"""Malformed/unknown metadata must fail, including under Python -O."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import inspect_native

from inspect_native import crc32c, inspect
from verify import verify_archive, verify_row

HERE = Path(__file__).resolve().parent


class MetadataGuardTests(unittest.TestCase):
    def setUp(self):
        self.row = copy.deepcopy(json.loads((HERE / 'excerpts.json').read_text())['files'][0])

    def mutate(self, name, edit, repair_crc=False):
        item = self.row['excerpts'][name]
        data = bytearray.fromhex(item['hex'])
        edit(data)
        if repair_crc:
            struct.pack_into('<I', data, 0, crc32c(b'\0' * 4 + data[4:]))
        item['hex'] = data.hex()
        item['sha256'] = hashlib.sha256(data).hexdigest()

    def test_valid(self):
        self.assertEqual(verify_row(self.row), (2000, 512))

    def test_header_crc(self):
        self.mutate('container_header', lambda b: b.__setitem__(40, b[40] ^ 1))
        with self.assertRaisesRegex(ValueError, 'Header CRC'):
            verify_row(self.row)

    def test_footer_crc(self):
        self.mutate('container_footer', lambda b: b.__setitem__(40, b[40] ^ 1))
        with self.assertRaisesRegex(ValueError, 'Footer CRC'):
            verify_row(self.row)

    def test_table_crc(self):
        self.mutate('segment_table', lambda b: b.__setitem__(4, b[4] ^ 1))
        with self.assertRaisesRegex(ValueError, 'Table CRC'):
            verify_row(self.row)

    def test_unknown_container_with_valid_crc(self):
        self.mutate('container_header', lambda b: struct.pack_into('<H', b, 6, 3), True)
        with self.assertRaisesRegex(ValueError, 'Unknown container'):
            verify_row(self.row)

    def test_unknown_quantizer(self):
        self.mutate('quantizer_fixed_prefix', lambda b: struct.pack_into('<H', b, 6, 99))
        with self.assertRaisesRegex(ValueError, 'Unknown quantizer'):
            verify_row(self.row)

    def test_disagreeing_chunk_count(self):
        self.mutate('quantizer_fixed_prefix', lambda b: struct.pack_into('<I', b, 28, 256))
        with self.assertRaisesRegex(ValueError, 'disagreement'):
            verify_row(self.row)

    def test_legacy_pq_header(self):
        self.mutate('diskann_pq_meta_header', lambda b: b.__setitem__(16, 1))
        with self.assertRaisesRegex(ValueError, 'Legacy/unknown'):
            verify_row(self.row)

    def test_unknown_metric(self):
        self.mutate('quantizer_fixed_prefix', lambda b: struct.pack_into('<I', b, 12, 99))
        with self.assertRaisesRegex(ValueError, 'Unknown quantizer'):
            verify_row(self.row)

    def test_document_code_arithmetic(self):
        self.mutate('diskann_meta_prefix', lambda b: struct.pack_into('<Q', b, 0, 2001))
        with self.assertRaisesRegex(ValueError, 'PQ code byte count'):
            verify_row(self.row)

    def test_flat_package_rejected(self):
        item = self.row['excerpts']['segment_table']
        data = bytearray.fromhex(item['hex'])
        data[:] = data.replace(b'diskann.meta', b'flatxxx.meta')
        item['hex'] = data.hex()
        item['sha256'] = hashlib.sha256(data).hexdigest()
        self.mutate('container_footer', lambda b: struct.pack_into('<I', b, 4, crc32c(data)), True)
        with self.assertRaisesRegex(ValueError, 'Unsupported non-DiskANN'):
            verify_row(self.row)

    def test_no_pq_interpretation_rejected(self):
        self.mutate('diskann_pq_meta_header', lambda b: struct.pack_into('<Q', b, 8, 0))
        with self.assertRaisesRegex(ValueError, 'Invalid PQ dimensions'):
            verify_row(self.row)

    def test_native_complete_metadata_crc(self):
        # Synthetic omitted content is zero-filled, never copied from the seed.
        import tempfile
        data = bytearray(self.row['bytes'])
        for excerpt in self.row['excerpts'].values():
            offset = excerpt['offset']
            data[offset:offset + excerpt['bytes']] = bytes.fromhex(excerpt['hex'])
        table_offset = self.row['excerpts']['segment_table']['offset']
        table_size = self.row['excerpts']['segment_table']['bytes']
        footer_offset = self.row['bytes'] - 128
        count = struct.unpack_from('<I', data, footer_offset + 12)[0]
        for index in range(count):
            base = table_offset + index * 32
            _, _, position, length, _ = struct.unpack_from('<IIQQQ', data, base)
            struct.pack_into('<I', data, base + 4, crc32c(data[64 + position:64 + position + length]))
        struct.pack_into('<I', data, footer_offset + 4, crc32c(data[table_offset:table_offset + table_size]))
        struct.pack_into('<I', data, footer_offset, crc32c(b'\0' * 4 + data[footer_offset + 4:]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic.index'
            path.write_bytes(data)
            expected = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            self.assertEqual(inspect(path, expected)['effective_pq_chunks'], 512)
            # Rebind whole-file hash after corruption so section CRC is tested.
            data[self.row['sections']['diskann.pq_meta']['offset'] + 204] ^= 1
            path.write_bytes(data)
            expected['sha256'] = hashlib.sha256(data).hexdigest()
            with self.assertRaises(ValueError):
                inspect(path, expected)

    def overlap_fixture(self, padding_only=False):
        # Metadata-only synthetic view: no omitted vectors/graph/PQ codes copied.
        row = copy.deepcopy(self.row)
        table = bytearray.fromhex(row['excerpts']['segment_table']['hex'])
        footer = bytearray.fromhex(row['excerpts']['container_footer']['hex'])
        count = struct.unpack_from('<I', footer, 12)[0]
        entries = {}
        for index in range(count):
            fields = list(struct.unpack_from('<IIQQQ', table, index*32))
            end = table.index(0, fields[0])
            entries[table[fields[0]:end].decode()] = (index, fields)
        sm = row['sections']['diskann.meta']
        sp = row['sections']['diskann.pq_meta']
        disk = bytes.fromhex(row['excerpts']['diskann_meta_prefix']['hex']) + bytes(sm['size']-80)
        pq = (bytes.fromhex(row['excerpts']['diskann_pq_meta_header']['hex'])
              + bytes.fromhex(row['excerpts']['quantizer_fixed_prefix']['hex'])
              + bytes(sp['size']-204))
        # Rebind metadata CRCs to the synthetic zero-filled omitted codebook.
        for name, data in [('diskann.meta', disk), ('diskann.pq_meta', pq)]:
            index, fields = entries[name]
            fields[1] = crc32c(data)
            struct.pack_into('<IIQQQ', table, index*32, *fields)
            row['sections'][name]['crc32c'] = fields[1]
        index, fields = entries['diskann.pq_data']
        meta = entries['diskann.pq_meta'][1]
        fields[2] = meta[2] + meta[3] + meta[4] - 1 if padding_only else meta[2]
        struct.pack_into('<IIQQQ', table, index*32, *fields)
        row['sections']['diskann.pq_data']['offset'] = 64 + fields[2]
        struct.pack_into('<I', footer, 4, crc32c(table))
        struct.pack_into('<I', footer, 0, crc32c(b'\0'*4 + footer[4:]))
        for name, data in [('segment_table', table), ('container_footer', footer)]:
            row['excerpts'][name]['hex'] = data.hex()
            row['excerpts'][name]['sha256'] = hashlib.sha256(data).hexdigest()
        blocks = [(item['offset'], bytes.fromhex(item['hex']))
                  for item in row['excerpts'].values()]
        blocks.extend([(sm['offset'], disk), (sp['offset'], pq)])
        class MetadataOnlyHandle:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def seek(self, offset): self.offset = offset
            def read(self, size):
                for start, data in blocks:
                    if start <= self.offset and self.offset+size <= start+len(data):
                        return data[self.offset-start:self.offset-start+size]
                raise AssertionError('omitted payload read forbidden')
        path = Mock()
        path.stat.return_value = SimpleNamespace(st_size=row['bytes'])
        path.open.return_value = MetadataOnlyHandle()
        expected = dict(bytes=row['bytes'], sha256='0'*64)
        return row, path, expected

    def test_crc_refreshed_overlap_rejected_by_both_readers(self):
        for padding_only in [False, True]:
            with self.subTest(padding_only=padding_only):
                row, path, expected = self.overlap_fixture(padding_only)
                with self.assertRaisesRegex(ValueError, '^Overlapping section extents$'):
                    verify_row(row)
                # Mock only whole-file identity; the real metadata parser/CRCs run.
                with patch.object(inspect_native, 'file_sha', return_value=expected['sha256']):
                    with self.assertRaisesRegex(ValueError, '^Overlapping section extents$'):
                        inspect(path, expected)

    def test_adjacent_and_empty_section_extents_valid(self):
        sections = {'second': dict(offset=12, size=4, padding=0),
                    'first': dict(offset=0, size=10, padding=2),
                    'empty': dict(offset=4, size=0, padding=0)}
        inspect_native.validate_section_extents(sections, 0, 16)

    def test_omitted_bytes_cannot_establish_full_file_identity(self):
        # A file with the recorded size but omitted bytes must fail the native
        # parser's whole-file provenance check, rather than accepting excerpts.
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic.index'
            with path.open('wb') as handle:
                handle.truncate(self.row['bytes'])
            with self.assertRaisesRegex(ValueError, 'differs from retained seed'):
                inspect(path, self.row)


class ArchiveGuardTests(unittest.TestCase):
    def test_top_level_command_accepts_retained_archive(self):
        import subprocess
        import sys
        result = subprocess.run([sys.executable, '-B', str(HERE/'verify.py')],
                                check=True, capture_output=True, text=True)
        report = json.loads(result.stdout)
        self.assertEqual((report['file_count'], report['documents'], report['effective_pq_chunks']),
                         (10, 18000, 512))

    def test_mutated_provenance_rejected_by_archive_entry(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            archive = parent/HERE.name
            archive.mkdir()
            (archive/'excerpts.json').write_bytes((HERE/'excerpts.json').read_bytes())
            provenance_dir = parent/'restart-floor-disk-2026-09-30'
            provenance_dir.mkdir()
            provenance = json.loads((HERE.parent/provenance_dir.name/'provenance.json').read_text())
            provenance['seed_documents'] = 17999
            (provenance_dir/'provenance.json').write_text(json.dumps(provenance))
            with self.assertRaisesRegex(ValueError, 'Provenance hash mismatch'):
                verify_archive(archive)
            # A refreshed pin must still fail the semantic corpus-count check.
            document = json.loads((archive/'excerpts.json').read_text())
            document['provenance_sha256'] = hashlib.sha256((provenance_dir/'provenance.json').read_bytes()).hexdigest()
            (archive/'excerpts.json').write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, 'Document count mismatch'):
                verify_archive(archive)


if __name__ == '__main__':
    unittest.main()
