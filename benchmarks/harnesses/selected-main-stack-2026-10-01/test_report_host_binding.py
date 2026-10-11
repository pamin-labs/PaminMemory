"""Packet-endpoint host reporting mocks; no product or current-host probes."""
import json
import unittest
from unittest.mock import patch
import test_templates as fixtures


class ReportHostBinding(unittest.TestCase):
    setUp = fixtures.Templates.setUp
    tearDown = fixtures.Templates.tearDown

    def packets(self):
        valid = {'actual_delta': {key: 1 for key in ['offered', 'scored', 'characters', 'tokens',
                 'padded_tokens', 'batches', 'encode_us', 'forward_us']},
                 'quality_at_actual_limit': {'recall': .5, 'mrr': .5, 'ndcg': .5},
                 'cpu_user_seconds': .01, 'cpu_system_seconds': .02}
        packets = []
        for job in self.common.schedule():
            actual = [{'arm': arm, 'wall_us': 100, 'process_after': {
                       'utime_ticks': 10, 'stime_ticks': 20, 'rss_kib': 30, 'hwm_kib': 40}}
                      for arm in job['sequence'].split(',')]
            packets.append({'job': job, 'rows': actual, 'validated': [valid] * len(actual),
                'opened': {'wall_us': 123}, 'usage': {'native_wall_seconds': 1, 'samples': []},
                'index_before_bytes': 1000, 'index_after_bytes': 1000,
                'host_conditions': {stage: {'host': {'cpu_model': 'packet CPU'},
                    'affinity': [0, 1], 'cpu_quota': '200000 100000'} for stage in ['before', 'after']}})
        return packets

    def report(self, packets):
        with patch.object(self.rows, 'validate', return_value=packets[0]['validated'][0]), \
             patch.object(self.rows, 'hot_guard'), patch.object(self.rows, 'oracle_checks', return_value=[]):
            return self.analyzer.report(packets)

    def test_edit_truncate_replace_or_remove_journal_cannot_change_report(self):
        packets = self.packets(); baseline = self.report(packets)
        journal = self.work / 'host-conditions.jsonl'
        for mutation in [json.dumps({'host': {'cpu_model': 'forged CPU'}}) + '\n', '', 'not JSON\n', None]:
            with self.subTest(mutation=mutation):
                if mutation is None:
                    journal.unlink()
                else:
                    journal.write_text(mutation)
                self.assertEqual(self.report(packets), baseline)
        self.assertEqual(len(baseline['host_conditions']), 160)
        self.assertEqual(baseline['host_conditions'][0], {
            'job': packets[0]['job']['name'], 'endpoint': 'before',
            'conditions': packets[0]['host_conditions']['before']})

    def test_missing_packet_endpoints_remain_na_despite_complete_journal(self):
        packets = self.packets(); packets[0].pop('host_conditions')
        result = self.report(packets)
        self.assertIsNone(result['host_conditions'][0]['conditions'])
        self.assertIsNone(result['host_conditions'][1]['conditions'])
        self.assertIn('missing endpoints N/A', result['host_conditions_scope'])
        cost = [row for row in result['metrics'] if row['metric'] in ['wall_us', 'rss_kib', 'hwm_kib']]
        self.assertTrue(cost)
        self.assertTrue(all(not row['stable_claim_eligible'] for row in cost))
        target = self.work / 'host-table.md'
        self.analyzer.write_tables(target, result)
        text = target.read_text()
        self.assertIn('Startup/monitoring journal is ancillary', text)
        self.assertIn('"conditions": null', text)

    def test_report_records_packet_endpoint_values_without_current_host_substitution(self):
        packets = self.packets(); packets[0]['host_conditions']['after']['host']['cpu_model'] = 'different endpoint CPU'
        with patch.object(self.common, 'guard', side_effect=AssertionError('current host forbidden')):
            result = self.report(packets)
        self.assertEqual(result['host_conditions'][1]['conditions']['host']['cpu_model'], 'different endpoint CPU')
        self.assertEqual(packets[0]['host_conditions']['before']['host']['cpu_model'], 'packet CPU')


if __name__ == '__main__':
    unittest.main()
