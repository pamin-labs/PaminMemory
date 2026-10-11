"""Pure metric eligibility fixtures; no model, native process or database."""
import copy
import importlib.machinery
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent


class HostTimingEligibility(unittest.TestCase):
    def setUp(self):
        loader = importlib.machinery.SourceFileLoader('metric_host_fixture', str(ROOT / 'metrics.py.in'))
        spec = importlib.util.spec_from_loader('metric_host_fixture', loader)
        self.metrics = importlib.util.module_from_spec(spec)
        loader.exec_module(self.metrics)
        self.observations = []
        self.conditions = {}
        self.provider = {'device': 'cpu', 'execution_provider': 'CPUExecutionProvider',
                         'scope': 'retained actual per-process log assignment/graph/runtime bindings revalidated by run.checkpoint'}
        for block in range(4):
            for source in ['main', 'stack']:
                name = f'{block}-{source}'
                job = {'name': name, 'tier': 'accurate', 'limit': 5, 'query_id': 101,
                       'sequence': 'A,B', 'source': source, 'round': block}
                self.observations.append({'job': job, 'phase': 'cold',
                    'row': {'arm': 'A', 'wall_us': 100 if source == 'main' else 90,
                            'process_after': {'rss_kib': 10, 'hwm_kib': 20}},
                    'validated': {'actual_delta': {k: 10 for k in ['offered', 'scored', 'characters',
                                   'tokens', 'padded_tokens', 'batches', 'encode_us', 'forward_us']},
                                  'quality_at_actual_limit': {'recall': .5, 'mrr': .5, 'ndcg': .5},
                                  'cpu_user_seconds': 1 if source == 'main' else .9,
                                  'cpu_system_seconds': 1 if source == 'main' else .9}})
                endpoint = {'host': {'architecture': 'x86_64', 'kernel': 'fixture-kernel', 'cpu_model': 'fixture CPU'},
                            'affinity': [0, 1], 'effective_cpu_quota_cores': 2,
                            'cgroup_ancestors': [{'path': '/fixture', 'memory_max_bytes': 8 * 1024**3,
                                                 'cpu_max': '200000 100000', 'memory_current_bytes': 1024}],
                            'captured_at_utc': 'volatile', 'disk_free': 9 * 1024**3,
                            'effective_memory_headroom_bytes': 8 * 1024**3 - 1024}
                self.conditions[name] = {'before': copy.deepcopy(endpoint), 'after': copy.deepcopy(endpoint)}

    def rows(self, conditions=None, provider=None):
        return self.metrics.metric_tables(self.observations, {},
                    self.conditions if conditions is None else conditions,
                    self.provider if provider is None else provider)

    def test_matching_positive_candidate_retains_timing_eligibility(self):
        wall = next(r for r in self.rows() if r['metric'] == 'wall_us')
        self.assertTrue(wall['stable_claim_eligible'])
        self.assertTrue(wall['cost_environment_eligible'])
        self.assertTrue(wall['accuracy_eligible'])

    def test_contradictory_fixed_endpoints_or_paired_blocks_withhold_timing_only(self):
        changes = [('host', 'cpu_model', 'different CPU'), ('host', 'kernel', 'different kernel'),
                   ('host', 'architecture', 'aarch64'), (None, 'affinity', [0]),
                   (None, 'effective_cpu_quota_cores', 1)]
        for nested, key, value in changes:
            for within_job in [False, True]:
                with self.subTest(key=key, within_job=within_job):
                    conditions = copy.deepcopy(self.conditions)
                    for phase in ['after'] if within_job else ['before', 'after']:
                        endpoint = conditions['2-stack'][phase]
                        (endpoint[nested] if nested else endpoint)[key] = value
                    wall = next(r for r in self.rows(conditions) if r['metric'] == 'wall_us')
                    self.assertFalse(wall['stable_claim_eligible'])
                    self.assertFalse(wall['cost_environment_eligible'])
                    self.assertTrue(wall['accuracy_eligible'])
        for field, value in [('memory_max_bytes', 4 * 1024**3), ('cpu_max', '400000 200000')]:
            conditions = copy.deepcopy(self.conditions)
            conditions['2-stack']['after']['cgroup_ancestors'][0][field] = value
            self.assertFalse(next(r for r in self.rows(conditions) if r['metric'] == 'wall_us')['stable_claim_eligible'])

    def test_volatile_memory_timestamp_and_free_space_do_not_require_exact_equality(self):
        conditions = copy.deepcopy(self.conditions)
        for index, record in enumerate(conditions.values()):
            record['after']['captured_at_utc'] = str(index)
            record['after']['disk_free'] += index
            record['after']['effective_memory_headroom_bytes'] -= index
            record['after']['cgroup_ancestors'][0]['memory_current_bytes'] += index
            record['after']['affinity'].reverse()
        self.assertTrue(next(r for r in self.rows(conditions) if r['metric'] == 'wall_us')['stable_claim_eligible'])

    def test_missing_or_invalid_host_and_provider_metadata_cannot_publish_timing(self):
        for conditions in [{}, {'0-main': {}}, copy.deepcopy(self.conditions)]:
            if len(conditions) == 8:
                conditions['0-main']['before']['effective_cpu_quota_cores'] = True
            row = next(r for r in self.rows(conditions) if r['metric'] == 'wall_us')
            self.assertFalse(row['stable_claim_eligible'])
            self.assertTrue(row['accuracy_eligible'])
        for provider in [{}, {'device': 'cpu'}, {**self.provider, 'execution_provider': 'CUDAExecutionProvider'}]:
            self.assertFalse(next(r for r in self.rows(provider=provider) if r['metric'] == 'wall_us')['stable_claim_eligible'])
        rows = self.metrics.metric_tables(self.observations, {})
        self.assertFalse(next(r for r in rows if r['metric'] == 'wall_us')['stable_claim_eligible'])
        self.assertTrue(next(r for r in rows if r['metric'] == 'quality_ndcg')['accuracy_eligible'])

    def test_rss_hwm_fixed_host_gate_withholds_memory_without_excluding_accuracy(self):
        for mode in ['matched', 'changed-memory-limit', 'missing']:
            with self.subTest(mode=mode):
                conditions = copy.deepcopy(self.conditions)
                if mode == 'changed-memory-limit':
                    conditions['2-stack']['after']['cgroup_ancestors'][0]['memory_max_bytes'] = 4 * 1024**3
                elif mode == 'missing':
                    conditions.pop('2-stack')
                memory = [row for row in self.rows(conditions) if row['metric'] in ['rss_kib', 'hwm_kib']]
                self.assertEqual(len(memory), 2)
                for row in memory:
                    self.assertEqual(row['stable_claim_eligible'], mode == 'matched')
                    self.assertEqual(row['cost_environment_eligible'], mode == 'matched')
                    self.assertTrue(row['accuracy_eligible'])
                    self.assertIn('RSS/HWM cost', row['cost_environment_scope'])

    def test_existing_table_renderer_reports_withheld_memory_eligibility(self):
        import sys
        import tempfile
        import types
        from unittest.mock import patch
        conditions = copy.deepcopy(self.conditions)
        conditions['2-stack']['after']['host']['cpu_model'] = 'different CPU'
        memory = [r for r in self.rows(conditions) if r['metric'] in ['rss_kib', 'hwm_kib']]
        loader = importlib.machinery.SourceFileLoader('memory_renderer_fixture', str(ROOT / 'analyze.py.in'))
        spec = importlib.util.spec_from_loader('memory_renderer_fixture', loader)
        analyzer = importlib.util.module_from_spec(spec)
        # Renderer-only import shims: report/checkpoint/native paths are not invoked.
        with patch.dict(sys.modules, {'common': types.ModuleType('common'),
                                     'rows': types.ModuleType('rows'), 'metrics': self.metrics}):
            loader.exec_module(analyzer)
        result = {'scope': 'fixture only', 'host_conditions': [{}], 'metrics': memory,
                  'correctness': [], 'control_diagnostics': [], 'process_metrics': [], 'disk': []}
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'table.md'
            analyzer.write_tables(target, result)
            text = target.read_text()
        self.assertEqual(text.count('descriptive; stable withheld'), 2)
        self.assertNotIn('descriptive four-block eligible', text)
