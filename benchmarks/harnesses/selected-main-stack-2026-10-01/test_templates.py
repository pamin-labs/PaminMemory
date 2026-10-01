"""Synthetic orchestration/arithmetic tests; never build or execute native code."""
import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

ROOT = Path(__file__).resolve().parent


def load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


class Templates(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.work = Path(self.temp.name)
        for path in ROOT.glob('*.py.in'):
            (self.work / path.name.removesuffix('.in')).write_bytes(path.read_bytes())
        manifest = json.loads((ROOT / 'sources.json').read_text())
        (self.work / 'sources.json').write_text(json.dumps(manifest))
        config = {key: str(self.work / key) for key in ['models', 'native', 'ort']}
        (self.work / 'config.json').write_text(json.dumps(config))
        self.common = load('common', self.work / 'common.py')
        self.metrics = load('metrics', self.work / 'metrics.py')
        self.rows = load('rows', self.work / 'rows.py')
        self.pg = load('owned_postgres', self.work / 'owned_postgres.py')
        self.run = load('reproduction_run', self.work / 'run.py')
        self.analyzer = load('reproduction_analyze', self.work / 'analyze.py')
        (self.work / 'main/crates/pamin-engine/tests/corpus').mkdir(parents=True)
        (self.work / 'main/crates/pamin-engine/tests/corpus/queries.json').write_text('[]')
        (self.work / 'build').mkdir()
        (self.work / 'build/binaries.json').write_text(json.dumps({'main': {'bytes': 10}, 'stack': {'bytes': 12}}))
        (self.work / 'seed-receipt.json').write_text(json.dumps({'index_allocated_bytes': 4096}))

    def tearDown(self):
        self.temp.cleanup()

    def test_post_start_faults_always_attempt_verified_cleanup(self):
        import subprocess
        item = {'home': self.work, 'data': self.work / 'data',
                'install': self.work / 'install', 'record': {'port': 1234, 'username': 'fixture', 'password': 'fixture', 'database': 'fixture'}}
        owned = {'pid': 42, 'port': 1234}
        for stage in ['launch', 'identity', 'sql', 'version', 'settings']:
            with self.subTest(stage=stage):
                def execute(command, **kwargs):
                    if '--version' in command:
                        return Mock(stdout=Path(command[0]).name + ' (PostgreSQL) 17.6')
                    if command[-1] == 'start' and stage == 'launch':
                        raise subprocess.TimeoutExpired(command, 35)
                    if command[-1] == 'SHOW server_version_num':
                        return Mock(stdout='170005' if stage == 'version' else '170006')
                    return Mock(stdout='wrong setting')
                with patch.object(self.pg, 'preflight', return_value=item), \
                     patch.object(self.pg.subprocess, 'run', side_effect=execute), \
                     patch.object(self.pg, 'identity', side_effect=RuntimeError('unowned') if stage == 'identity' else None, return_value=owned), \
                     patch.object(self.pg, 'listening', return_value=True), \
                     patch.object(self.pg, 'show_data_directory', side_effect=subprocess.TimeoutExpired('SQL', 5) if stage == 'sql' else None), \
                     patch.object(self.pg, 'stop_item') as stop:
                    with self.assertRaises((RuntimeError, subprocess.TimeoutExpired)):
                        self.pg.start(self.work, {})
                    stop.assert_called_once_with(item, {}, None if stage in ['launch', 'identity'] else owned)

    def test_unproven_seed_pid_never_runs_stop_command(self):
        home = self.work / 'seed-owned'
        data = home / 'postgres/data'
        data.mkdir(parents=True)
        (home / '.graph-disposable-clone').touch()
        (data / 'PG_VERSION').write_text('17')
        (data / 'postmaster.pid').write_text('42\n' + str(data) + '\n0\n1234\n')
        install = home / 'postgres/install/17.6.0'
        with patch.object(self.pg, 'identity', side_effect=RuntimeError('unowned PID')), \
             patch.object(self.pg.subprocess, 'run') as execute:
            with self.assertRaisesRegex(RuntimeError, 'unowned PID'):
                self.pg.stop_seed(home, {}, install)
            execute.assert_not_called()
        self.assertTrue((data / 'postmaster.pid').exists())

    def test_native_start_failure_does_not_guess_stop_identity(self):
        with patch.object(self.pg, 'start', side_effect=RuntimeError('start refused')), \
             patch.object(self.pg, 'stop') as stop, \
             patch.object(self.run.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError, 'start refused'):
                self.run.native(self.work, {}, 'unused', self.work / 'log')
            stop.assert_not_called()
            launch.assert_not_called()

    def test_probe_is_identical_to_measured_source(self):
        manifest = json.loads((ROOT / 'sources.json').read_text())
        self.assertEqual(hashlib.sha256((ROOT / 'probe.rs.in').read_bytes()).hexdigest(), manifest['probe_sha256'])

    def test_matrix_rotation_and_every_actual_call(self):
        jobs = self.common.schedule()
        self.assertEqual(len(jobs), 80)
        self.assertEqual(len({j['name'] for j in jobs}), 80)
        self.assertEqual(sum(len(j['sequence'].split(',')) for j in jobs), 880)
        for round_ in range(4):
            block = [j for j in jobs if j['round'] == round_]
            self.assertEqual(sum(j['source'] == 'main' for j in block), 10)
        off = next(j for j in jobs if j['tier'] == 'off')
        self.assertEqual(self.common.phase(off, 2), 'off-new-query')

    def test_quantile_and_variability_withhold_speed_claims(self):
        self.assertEqual(self.metrics.percentile([1, 2, 3, 4, 5], .95), 4.8)
        evidence = self.metrics.stability([1, 1, 1, 1], True, True,
                                         {'main': [1, 2, 1, 2], 'stack': [2, 3, 2, 3]})
        self.assertFalse(evidence['stable_claim_eligible'])
        self.assertIsNone(self.metrics.difference(0, 1)['percentage_change'])

    def test_missing_calls_fail_before_any_report(self):
        with self.assertRaises(ValueError):
            self.analyzer.report([])

    def test_all_metric_categories_from_complete_mock_product_rows(self):
        # Deliberately mocked product validation: these rows test orchestration,
        # units and completeness, not the model, retrieval or exact oracle.
        valid = {'actual_delta': {key: 1 for key in ['offered', 'scored', 'characters', 'tokens',
                  'padded_tokens', 'batches', 'encode_us', 'forward_us']},
                 'quality_at_actual_limit': {'recall': .5, 'mrr': .5, 'ndcg': .5},
                 'cpu_user_seconds': .01, 'cpu_system_seconds': .02}
        packets = []
        for job in self.common.schedule():
            snapshots = {'utime_ticks': 10, 'stime_ticks': 20, 'rss_kib': 30, 'hwm_kib': 40}
            actual = [{'arm': arm, 'wall_us': 100 if job['source'] == 'main' else 110,
                       'process_after': snapshots} for arm in job['sequence'].split(',')]
            packets.append({'job': job, 'rows': actual, 'validated': [valid] * len(actual),
                            'opened': {'wall_us': 123}, 'usage': {'native_wall_seconds': 1, 'samples': []},
                            'index_before_bytes': 1000, 'index_after_bytes': 1000})
        classifications = [{'job': packet['job'], 'paired_speed_claim_eligible': False,
                            'same_final_actual_inputs': True, 'same_final_raw_bits/order': False,
                            'all_changed_and_hot_exact': False, 'legacy_main_context_failure': True,
                            'retrieval_or_input_context_mismatch': False}
                           for packet in packets if packet['job']['tier'] == 'accurate']
        with patch.object(self.rows, 'validate', return_value=valid), \
             patch.object(self.rows, 'hot_guard'), \
             patch.object(self.rows, 'oracle_checks', return_value=classifications):
            result = self.analyzer.report(packets)
            self.assertEqual(result['calls'], 880)
            self.assertEqual(len(result['correctness']), 40)
            categories = {row['metric'] for row in result['metrics']}
            self.assertTrue({'wall_us', 'quality_ndcg', 'rss_kib', 'cpu_user_seconds', 'forward_us'} <= categories)
            changed = [r for r in result['metrics'] if r['configuration'][4] == 'changed-hot']
            self.assertTrue(all(not r['accuracy_eligible'] for r in changed))
            hot = [r for r in result['metrics'] if r['configuration'][4] == 'initial-hot']
            self.assertTrue(all(r['samples_per_arm'] == 20 for r in hot))
            self.assertTrue(any(r['metric'] == 'full_lifetime_cpu_seconds' and r['before'] is None
                                for r in result['process_metrics']))
            missing = [dict(p) for p in packets]
            missing[0]['rows'] = missing[0]['rows'][:-1]
            with self.assertRaises(ValueError):
                self.analyzer.report(missing)

    def test_materializer_refuses_original_repo_or_existing_output(self):
        module = load('materializer', ROOT / 'materialize.py')
        with self.assertRaises(ValueError):
            module.materialize(self.work, self.work / 'inside-repo', {})
        with self.assertRaises(ValueError):
            module.materialize(self.work, self.work, {})

    def test_frozen_source_archive_checks_every_selected_input(self):
        module = load('materializer', ROOT / 'materialize.py')
        identity = json.loads((ROOT / 'sources.json').read_text())['stack_source_subset']
        destination = self.work / 'stack-export'
        module.export_stack(destination, identity)
        manifest = json.loads((ROOT / identity['manifest']).read_text())
        self.assertEqual(set(module.inventory(destination)), set(manifest['files']))
        self.assertTrue((destination / 'LICENSE').is_file())
        self.assertTrue((destination / 'NOTICE').is_file())
        self.assertFalse((destination / 'internal-docs').exists())
        self.assertFalse((destination / '.git').exists())

    def test_changed_materialized_compilation_input_is_rejected(self):
        source = self.work / 'stack'
        source.mkdir()
        path = source / 'Cargo.toml'
        path.write_text('immutable source')
        receipt = {'stack': {'Cargo.toml': {'sha256': self.common.digest(path)}}}
        (self.work / 'source-receipts.json').write_text(json.dumps(receipt))
        self.common.source_inputs()
        path.write_text('changed source')
        with self.assertRaises(ValueError):
            self.common.source_inputs()

    def test_actual_provider_gate_rejects_unbound_gpu_execution(self):
        job = self.common.schedule()[0]
        raw = [{'loaded_libraries': ['1-2 r-xp 0 00:00 1 ' + str(self.work / 'native/libzvec_c_api.so'),
                                    '1-2 r-xp 0 00:00 1 ' + str(self.work / 'ort/libonnxruntime.so.1.28.0')]}
               for _ in job['sequence'].split(',')]
        text = 'test result: ok. 1 passed; 0 failed;\n'
        text += 'CACHE_ENGINE_OPEN {"documents":230,"coverage":0.0}\n'
        text += 'CACHE_ENGINE_GRAPH [["mentions",11]]\n'
        text += ''.join('CACHE_ENGINE_ROW ' + json.dumps(row) + '\n' for row in raw)
        bound = {}
        for role, pin in self.common.MANIFEST['roles'].items():
            path = str(self.work / 'models' / pin['relative'])
            bound[path] = {'sha256': 'synthetic'}
            text += 'ONNX graph execution-provider assignment model_graph=' + json.dumps(path)
            text += ' assigned_nodes=' + json.dumps({'CPUExecutionProvider': pin['nodes']}) + '\n'
        with patch.object(self.common, 'digest', return_value='synthetic'):
            self.run.parse(text, job, bound)
            with self.assertRaises(ValueError):
                self.run.parse(text.replace('CPUExecutionProvider', 'CUDAExecutionProvider'), job, bound)


if __name__ == '__main__':
    unittest.main()
