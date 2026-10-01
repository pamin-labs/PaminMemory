"""Invariant negatives mutate metadata and refresh CRCs, not just file hashes."""
import copy
import hashlib
import json
import unittest

import verify


class MetadataInvariants(unittest.TestCase):
    def setUp(self):
        self.evidence = json.loads((verify.ROOT/'evidence.json').read_text())

    def reject(self, mutation):
        changed = copy.deepcopy(self.evidence)
        mutation(changed)
        with self.assertRaises(ValueError):
            verify.verify(changed, check_logs=False)

    def rewrite(self, evidence, section, fields):
        excerpt = evidence['runs'][0]['files'][0]['excerpts']['after']
        layout = verify.HEADER if section == 'header_hex' else verify.FOOTER
        values = list(layout.unpack(bytes.fromhex(excerpt[section])))
        for index, value in fields.items(): values[index] = value
        values[0] = 0
        values[0] = verify.crc32c(layout.pack(*values))
        excerpt[section] = layout.pack(*values).hex()

    def test_valid(self):
        self.assertTrue(verify.verify(self.evidence))

    def test_invalid_crc(self):
        def mutate(e):
            x = e['runs'][0]['files'][0]['excerpts']['after']
            x['footer_hex'] = '00'*4 + x['footer_hex'][8:]
        self.reject(mutate)

    def test_rehashed_wrong_format(self):
        self.reject(lambda e:self.rewrite(e, 'header_hex', {2:3}))

    def test_crc_refreshed_wrong_magic_each_file_both_phases(self):
        pins = list(verify.HEADER_MAGIC.values())
        for run_index in range(2):
            for file_index in range(4):
                # Zero and another file's legitimate identity must both fail.
                for magic in [0, pins[(file_index+1) % 4]]:
                    with self.subTest(arm=run_index, file=file_index, magic=magic):
                        changed = copy.deepcopy(self.evidence)
                        file = changed['runs'][run_index]['files'][file_index]
                        for excerpt in file['excerpts'].values():
                            values = list(verify.HEADER.unpack(bytes.fromhex(excerpt['header_hex'])))
                            values[4] = magic; values[0] = 0
                            values[0] = verify.crc32c(verify.HEADER.pack(*values))
                            excerpt['header_hex'] = verify.HEADER.pack(*values).hex()
                        with self.assertRaisesRegex(ValueError, 'header magic identity receipt binding'):
                            verify.verify(changed, check_logs=False)

    def test_wrong_linear_header_meta_extent_all_excerpts(self):
        # Both phases agree and the 64-vector counts remain unchanged.
        for extent in [1, 63, 65]:
            with self.subTest(extent=extent):
                changed = copy.deepcopy(self.evidence)
                for run in changed['runs']:
                    for file in run['files']:
                        for excerpt in file['excerpts'].values():
                            data = bytearray.fromhex(excerpt['streamer_and_linear_header_hex'])
                            values = list(verify.LINEAR.unpack_from(data, 64))
                            values[7] = values[0] - extent
                            verify.LINEAR.pack_into(data, 64, *values)
                            excerpt['streamer_and_linear_header_hex'] = data.hex()
                with self.assertRaisesRegex(ValueError, 'linear header bounds'):
                    verify.verify(changed, check_logs=False)

    def test_rehashed_table_bounds(self):
        self.reject(lambda e:self.rewrite(e, 'footer_hex', {4:2**30}))

    def test_rehashed_file_bounds(self):
        self.reject(lambda e:self.rewrite(e, 'footer_hex', {6:2**30}))

    def test_rehashed_chained_format(self):
        self.reject(lambda e:self.rewrite(e, 'footer_hex', {len(verify.FOOTER.unpack(bytes(128)))-2:64}))

    def test_wrong_declared_storage_type(self):
        self.reject(lambda e:e.update(storage_type='VectorFp32'))

    def test_wrong_linear_count(self):
        def mutate(e):
            x = e['runs'][0]['files'][0]['excerpts']['after']
            data = bytearray.fromhex(x['streamer_and_linear_header_hex'])
            data[68:72] = (65).to_bytes(4,'little')
            x['streamer_and_linear_header_hex'] = data.hex()
        self.reject(mutate)

    def test_unallowed_metadata_byte_with_valid_footer_crc(self):
        self.reject(lambda e:self.rewrite(e, 'footer_hex', {8:1}))

    def test_allowed_mask_range_expansion(self):
        self.reject(lambda e:e['runs'][0]['files'][0]['allowed_full_field_ranges'][0].__setitem__(1,1052592))

    def test_late_phase_fabrication(self):
        self.reject(lambda e:e['runs'][1]['phase_delta_file_counts'].update(vector_queried=4))

    def test_native_binding(self):
        self.reject(lambda e:e.update(native_sha256='0'*64))

    def test_rehashed_wrong_segment_type(self):
        def mutate(e):
            x = e['runs'][0]['files'][0]['excerpts']['after']
            table = bytearray(x['table_size'])
            for offset, text in x['table_nonzero_runs']:
                data = bytes.fromhex(text); table[offset:offset+len(data)] = data
            offset = table.index(b'flat.linear_meta')
            table[offset] = ord('X')
            runs = []; start = None
            for i, byte in enumerate(table + b'\0'):
                if byte and start is None: start = i
                elif not byte and start is not None:
                    runs.append([start, table[start:i].hex()]); start = None
            x['table_nonzero_runs'] = runs
            self.rewrite(e, 'footer_hex', {1:verify.crc32c(table)})
        self.reject(mutate)

    def test_helper_source_binding(self):
        self.reject(lambda e:e.update(helper_source_sha256='0'*64))

    def test_controller_binding(self):
        self.reject(lambda e:e.update(controller_sha256='0'*64))

    def test_changed_runtime_receipt(self):
        self.reject(lambda e:e['runtime_assets']['libzvec_c_api.so'].update(sha256='0'*64))

    def test_changed_source_receipt(self):
        self.reject(lambda e:next(iter(e['source_files'].values())).update(sha256='0'*64))

    def test_broadened_public_proof_scope(self):
        self.reject(lambda e:e['runs'][0]['private_full_byte_verification'].update(scope='public full payload proof'))

    def test_broadened_component_scope(self):
        self.reject(lambda e:e['runs'][0]['environment'].update(inference='Engine/model'))

    def test_changed_run_start(self):
        self.reject(lambda e:e['runs'][0].update(started_utc='2026-10-01T01:02:05.865611+00:00'))

    def test_changed_observation_receipt(self):
        self.reject(lambda e:e['runs'][0]['snapshot_observation_utc'].update(opened='2026-10-01T01:02:06.233628+00:00'))

    def reject_rehashed_log(self, arm, replace):
        changed = copy.deepcopy(self.evidence)
        logs = {r['arm']:(verify.ROOT/(r['arm']+'.log')).read_text() for r in changed['runs']}
        logs[arm] = replace(logs[arm])
        row = next(r for r in changed['runs'] if r['arm'] == arm)
        row['log_sha256'] = hashlib.sha256(logs[arm].encode()).hexdigest()
        with self.assertRaises(ValueError):
            verify.verify(changed, logs=logs)

    def test_rehashed_log_false_read_only_getter(self):
        self.reject_rehashed_log('vector', lambda log:log.replace('read_only=true','read_only=false'))

    def test_rehashed_log_duplicate_option_getter(self):
        self.reject_rehashed_log('pure-open', lambda log:log+'PROBE_OPTIONS read_only=true enable_mmap=true max_buffer_size=67108864\n')

    def test_rehashed_log_duplicate_hits(self):
        self.reject_rehashed_log('vector', lambda log:log+'PROBE_HITS 50\n')

    def test_rehashed_log_pure_open_hits(self):
        self.reject_rehashed_log('pure-open', lambda log:log+'PROBE_HITS 50\n')

    def change_table_entries(self, evidence, mutations):
        # Mutate both phases consistently, then refresh table and footer CRCs.
        for phase in ['before', 'after']:
            excerpt = evidence['runs'][0]['files'][0]['excerpts'][phase]
            table = bytearray(excerpt['table_size'])
            for offset, text in excerpt['table_nonzero_runs']:
                data = bytes.fromhex(text); table[offset:offset+len(data)] = data
            for index, fields in mutations.items():
                values = list(verify.SEGMENT.unpack_from(table,index*32))
                for field, value in fields.items():values[field] = value
                verify.SEGMENT.pack_into(table,index*32,*values)
            runs = []; start = None
            for i, byte in enumerate(table + b'\0'):
                if byte and start is None:start = i
                elif not byte and start is not None:
                    runs.append([start,table[start:i].hex()]);start = None
            excerpt['table_nonzero_runs'] = runs
            values = list(verify.FOOTER.unpack(bytes.fromhex(excerpt['footer_hex'])))
            values[1] = verify.crc32c(table);values[0] = 0
            values[0] = verify.crc32c(verify.FOOTER.pack(*values))
            excerpt['footer_hex'] = verify.FOOTER.pack(*values).hex()

    def test_rehashed_padding_overflow_both_phases(self):
        self.reject(lambda e:self.change_table_entries(e,{0:{4:2**63}}))

    def test_rehashed_segment_overlap_both_phases(self):
        self.reject(lambda e:self.change_table_entries(e,{1:{2:4095}}))

    def test_rehashed_segment_gap_both_phases(self):
        self.reject(lambda e:self.change_table_entries(e,{1:{2:4097}}))

    def test_rehashed_aggregate_padding_gap_both_phases(self):
        self.reject(lambda e:self.change_table_entries(e,{4:{4:0}}))

    def test_rehashed_segment_reordering_both_phases(self):
        self.reject(lambda e:self.change_table_entries(e,{0:{2:4096},1:{2:0}}))

    def test_all_whole_file_identity_receipts_bound(self):
        # Mutate every one of the sixteen receipts; metadata CRCs stay valid.
        for run_index in range(2):
            for file_index in range(4):
                for field in ['whole_file_sha256_before', 'whole_file_sha256_after']:
                    with self.subTest(arm=run_index, file=file_index, field=field):
                        changed = copy.deepcopy(self.evidence)
                        changed['runs'][run_index]['files'][file_index][field] = '0'*64
                        with self.assertRaisesRegex(ValueError, 'whole-file identity receipt binding'):
                            verify.verify(changed, check_logs=False)

    def test_malformed_whole_file_identity_receipts(self):
        for value in [None, 123, 'x'*64, '0'*63, 'A'*64]:
            with self.subTest(value=value):
                changed = copy.deepcopy(self.evidence)
                changed['runs'][0]['files'][0]['whole_file_sha256_before'] = value
                with self.assertRaisesRegex(ValueError, 'whole-file identity format'):
                    verify.verify(changed, check_logs=False)

    def test_rehashed_log_concatenated_failed_summary(self):
        self.reject_rehashed_log('vector', lambda log:log+'test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.0s\n')

    def test_rehashed_log_duplicate_positive_summary(self):
        self.reject_rehashed_log('pure-open', lambda log:log+'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 3.62s\n')

    def test_rehashed_log_concatenated_zero_test_summary(self):
        self.reject_rehashed_log('vector', lambda log:log+'test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.0s\n')

    def test_rehashed_log_nonzero_ignored_count(self):
        self.reject_rehashed_log('pure-open', lambda log:log.replace('0 ignored','1 ignored'))

    def test_rehashed_log_missing_summary(self):
        self.reject_rehashed_log('vector', lambda log:'\n'.join(line for line in log.splitlines() if not line.startswith('test result:')))

    def test_crc_refreshed_unsupported_revision_both_phases(self):
        def mutate(e):
            for phase in ['before', 'after']:
                excerpt = e['runs'][0]['files'][0]['excerpts'][phase]
                values = list(verify.HEADER.unpack(bytes.fromhex(excerpt['header_hex'])))
                values[3] = 1; values[0] = 0
                values[0] = verify.crc32c(verify.HEADER.pack(*values))
                excerpt['header_hex'] = verify.HEADER.pack(*values).hex()
        self.reject(mutate)

    def test_rehashed_options_after_open(self):
        option = 'PROBE_OPTIONS read_only=true enable_mmap=true max_buffer_size=67108864\n'
        for arm in ['pure-open', 'vector']:
            with self.subTest(arm=arm):
                self.reject_rehashed_log(arm, lambda log:log.replace(option,'').replace('PROBE_STAGE opened\n','PROBE_STAGE opened\n'+option))

    def test_rehashed_options_after_drop(self):
        option = 'PROBE_OPTIONS read_only=true enable_mmap=true max_buffer_size=67108864\n'
        self.reject_rehashed_log('vector', lambda log:log.replace(option,'').replace('PROBE_STAGE dropped\n','PROBE_STAGE dropped\n'+option))

    def test_rehashed_hits_after_drop(self):
        self.reject_rehashed_log('vector', lambda log:log.replace('PROBE_HITS 50\n','').replace('PROBE_STAGE dropped\n','PROBE_STAGE dropped\nPROBE_HITS 50\n'))

    def test_rehashed_hits_before_preparation(self):
        self.reject_rehashed_log('vector', lambda log:log.replace('PROBE_HITS 50\n','').replace('PROBE_STAGE vector_prepared\n','PROBE_HITS 50\nPROBE_STAGE vector_prepared\n'))

    def test_unverified_file_claim(self):
        for arm in range(2):
            for index in range(4):
                with self.subTest(arm=arm, file=index):
                    self.reject(lambda e:e['runs'][arm]['files'][index].update(public_full_payload_verified=True))

    def test_unverified_excerpts_claim(self):
        self.reject(lambda e:e['runs'][0]['files'][0]['excerpts'].update(public_full_payload_verified=True))

    def test_unverified_phase_excerpt_claim(self):
        for phase in ['before', 'after']:
            with self.subTest(phase=phase):
                self.reject(lambda e:e['runs'][0]['files'][0]['excerpts'][phase].update(public_full_payload_verified=True))

    def test_source_binding(self):
        self.reject(lambda e:e.update(vendor_commit='0'*40))


if __name__ == '__main__':
    unittest.main()
