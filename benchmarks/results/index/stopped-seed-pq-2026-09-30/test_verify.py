"""Malformed/unknown metadata must fail, including under Python -O."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import unittest

from inspect_native import crc32c, inspect
from verify import verify_row

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


if __name__ == '__main__':
    unittest.main()
