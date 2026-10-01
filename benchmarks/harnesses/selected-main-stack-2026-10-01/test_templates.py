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
        (self.work / 'host-conditions.jsonl').write_text(json.dumps({'cpu_quota': '400000 100000', 'affinity': [0, 1, 2, 3], 'fixture': True}) + '\n')
        (self.work / 'seed-receipt.json').write_text(json.dumps({'index_allocated_bytes': 4096}))

    def tearDown(self):
        self.temp.cleanup()

    def mock_cgroup(self):
        root = self.work / 'cgroup'
        leaf = root / 'slice/pod'
        leaf.mkdir(parents=True)
        for node, maximum, current, quota in [(root, 'max', 9, 'max 100000'),
                (root / 'slice', 12 * 1024**3, 11 * 1024**3, '200000 100000'),
                (leaf, 8 * 1024**3, 1024**3, '500000 100000')]:
            (node / 'memory.max').write_text(str(maximum))
            (node / 'memory.current').write_text(str(current))
            (node / 'cpu.max').write_text(quota)
        membership = self.work / 'membership'
        membership.write_text('0::/slice/pod\n')
        return root, leaf, membership

    def test_nested_ancestor_headroom_and_cpu_limits(self):
        root, leaf, membership = self.mock_cgroup()
        value = self.common.cgroup_conditions(root, membership)
        self.assertEqual(value['cgroup_path'], str(leaf))
        self.assertEqual(value['cgroup_memory_bytes'], 1024**3)
        self.assertEqual(value['effective_memory_headroom_bytes'], 1024**3)
        self.assertEqual(value['effective_cpu_quota_cores'], 2)
        with patch.object(self.common, 'cgroup_conditions', return_value=value), \
             patch.object(self.common.shutil, 'disk_usage', return_value=Mock(free=self.common.RESERVE + 1)):
            with self.assertRaisesRegex(ValueError, '2 GiB'):
                self.common.guard('nested-fixture')
        # Failure still retains the observation rather than losing conditions.
        retained = json.loads((self.work / 'host-conditions.jsonl').read_text().splitlines()[-1])
        self.assertEqual(retained['phase'], 'nested-fixture')
        self.assertEqual(retained['effective_cpu_quota_cores'], 2)
        self.assertIn('architecture', retained['host'])
        (root / 'slice/memory.current').write_text(str(2 * 1024**3))
        value = self.common.cgroup_conditions(root, membership)
        self.assertEqual(value['effective_memory_headroom_bytes'], 7 * 1024**3)

    def test_cgroup_path_escapes_and_symlinks_are_refused(self):
        root, leaf, membership = self.mock_cgroup()
        for escape in ['/../outside', '/slice/../../outside', 'relative']:
            membership.write_text('0::' + escape + '\n')
            with self.assertRaisesRegex(ValueError, 'path escape'):
                self.common.cgroup_conditions(root, membership)
        membership.write_text('0::/missing-leaf\n')
        with self.assertRaisesRegex(ValueError, 'effective cgroup missing'):
            self.common.cgroup_conditions(root, membership)
        outside = self.work / 'outside'; outside.mkdir()
        (root / 'escape').symlink_to(outside, target_is_directory=True)
        membership.write_text('0::/escape\n')
        with self.assertRaisesRegex(ValueError, 'path escape'):
            self.common.cgroup_conditions(root, membership)
        (root / 'alias').symlink_to(leaf, target_is_directory=True)
        membership.write_text('0::/alias\n')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.common.cgroup_conditions(root, membership)

    def test_owned_pg_disk_counts_local_wal_and_excludes_external_links(self):
        data = self.work / 'postgres/data'; wal = data / 'pg_wal'; wal.mkdir(parents=True)
        (data / 'base-file').write_bytes(b'base')
        (wal / 'wal-file').write_bytes(b'wal')
        outside = self.work / 'unowned'; outside.mkdir(); (outside / 'ignored').write_bytes(b'external')
        (data / 'external').symlink_to(outside, target_is_directory=True)
        (data / 'linked-file').symlink_to(outside / 'ignored')
        value = self.common.disk_sizes(data)
        self.assertEqual(value['logical_bytes'], 7)
        self.assertEqual(value['files'], 2)
        self.assertEqual(value['allocated_bytes'], sum(p.stat().st_blocks * 512 for p in [data/'base-file', wal/'wal-file']))

    def test_public_hot_samples_are_four_off_and_twenty_accurate(self):
        root = ROOT.parents[1] / 'results/inference/selected-main-stack-2026-10-01'
        verifier = load('public_sample_verifier', root / 'verify.py')
        data = json.loads((root / 'evidence.json').read_text())
        result = verifier.tables(data)
        off = [r for r in result['metrics'] if r['configuration'][4] == 'off-hot']
        accurate = [r for r in result['metrics'] if r['configuration'][4] in ['initial-hot', 'changed-hot']]
        self.assertTrue(off and accurate)
        self.assertTrue(all(r['samples_per_arm'] == 4 for r in off))
        self.assertTrue(all(r['samples_per_arm'] == 20 for r in accurate))
        markdown = verifier.markdown(result)
        self.assertIn('Samples per arm', markdown)
        self.assertIn('Off hot quantiles pool four calls', markdown)

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

    def test_changed_history_requires_actual_input_bytes_to_change(self):
        import copy
        results = {}
        for source in ['main', 'stack']:
            for initial in ['A', 'B']:
                final = 'B' if initial == 'A' else 'A'
                job = {'round': 0, 'source': source, 'limit': 5, 'query_id': 80,
                       'tier': 'accurate', 'sequence': initial + ',' + final}
                def row(arm):
                    return {'query': 'query', 'fused': [{'topic_id': arm, 'topic': 'topic',
                            'state': {'content': arm}, 'seed': None}],
                            'selected_fused_positions': [0], 'complete': [arm], 'limited': [arm]}
                actual = [row(initial) for _ in range(6)] + [row(final) for _ in range(6)] + [row('N')]
                results[(0, source, 5, 80, 'accurate', initial)] = {'job': job, 'rows': actual}
        with patch.object(self.rows, 'typed_hits', side_effect=lambda value: json.dumps(value)), \
             patch.object(self.rows, 'delta', return_value={'scored': 1}):
            classifications = self.rows.oracle_checks(results)
            unchanged = copy.deepcopy(results)
            # IDs and oracle contexts differ, but the actual ordered model bytes do not.
            for record in unchanged.values():
                for row in record['rows']:
                    row['fused'][0]['state']['content'] = 'identical document'
            with self.assertRaisesRegex(AssertionError, 'changed-history.*input'):
                self.rows.oracle_checks(unchanged)
            self.assertTrue(all(c['changed_actual_reranker_inputs'] for c in classifications))

    def test_native_zombie_memory_race_skips_sample_and_stops_owned_pg(self):
        owned = {'pid': 42}
        process = Mock(pid=1234)
        def proc_status(path, *args, **kwargs):
            return 'Name:\thelper\nState:\tZ (zombie)\n'
        with patch.object(self.pg, 'start', return_value={'identity': owned}), \
             patch.object(self.pg, 'check'), patch.object(self.pg, 'stop') as stop, \
             patch.object(self.common, 'guard'), patch.object(self.common, 'stop_group'), \
             patch.object(self.common, 'exit_status', side_effect=[None, 0]), \
             patch.object(self.run.subprocess, 'Popen', return_value=process), \
             patch.object(self.run.time, 'sleep'), patch.object(Path, 'read_text', proc_status):
            usage = self.run.native(self.work, {}, 'unused', self.work / 'zombie')
        self.assertEqual(usage['samples'], [])
        stop.assert_called_once_with(self.work, {}, owned)
        with patch.object(Path, 'read_text', return_value='VmRSS:\t12 kB\nVmHWM:\t34 kB\n'):
            self.assertEqual(self.run.status(1234), {'VmRSS': 12, 'VmHWM': 34})

    def test_pinned_toolchain_requires_exact_version_token(self):
        module = load('reproduction_build', self.work / 'build.py')
        for tool in ['cargo', 'rustc', 'rustdoc']:
            self.common.CONFIG[tool] = 'mock-' + tool
        def versions(command, **kwargs):
            return command[0].removeprefix('mock-') + ' 1.98.1 (fixture 2026-10-01)\n'
        with patch.object(module.subprocess, 'check_output', side_effect=versions):
            receipt = module.toolchain_versions()
        self.assertEqual(receipt['rustc']['version'], '1.98.1')
        self.assertIn('fixture', receipt['rustc']['reported'])
        for invalid in ['1.98.10', '1.98.1-nightly', '1.98.0', 'prefix 1.98.1']:
            with patch.object(module.subprocess, 'check_output', return_value='rustc ' + invalid + '\n'):
                with self.assertRaisesRegex(ValueError, 'pinned toolchain'):
                    module.toolchain_versions()

    def test_native_start_failure_does_not_guess_stop_identity(self):
        with patch.object(self.pg, 'start', side_effect=RuntimeError('start refused')), \
             patch.object(self.pg, 'stop') as stop, \
             patch.object(self.run.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError, 'start refused'):
                self.run.native(self.work, {}, 'unused', self.work / 'log')
            stop.assert_not_called()
            launch.assert_not_called()

    def test_timeout_terminates_entire_new_owned_group(self):
        import signal
        process = Mock(pid=1234)
        process.poll.return_value = None
        # Model an owned leader plus compiler/helper descendants and an unrelated group.
        groups = {1234: {1234, 1235, 1236}, 9999: {9999}}
        def terminate(group, sig):
            self.assertEqual(sig, signal.SIGKILL)
            groups.pop(group)
        with patch.object(self.common.subprocess, 'Popen', return_value=process) as launch, \
             patch.object(self.common, 'guard'), \
             patch.object(self.common, 'exit_status', return_value=None), \
             patch.object(self.common, 'live_group_members', return_value=[]), \
             patch.object(self.common.os, 'killpg', side_effect=terminate):
            with self.assertRaisesRegex(ValueError, 'timeout'):
                self.common.monitored(['mock helper'], self.work, {}, self.work / 'fault', limit=0)
            self.assertTrue(launch.call_args.kwargs['start_new_session'])
            process.wait.assert_called_once_with(timeout=10)
        self.assertEqual(groups, {9999: {9999}})
        self.assertTrue((self.work / 'fault.stdout').exists())
        self.assertTrue((self.work / 'fault.stderr').exists())

    def test_exited_parent_still_cleans_descendant_group(self):
        import signal
        process = Mock(pid=1234, returncode=1)
        process.poll.return_value = 1
        with patch.object(self.common.subprocess, 'Popen', return_value=process), \
             patch.object(self.common, 'exit_status', return_value=1), \
             patch.object(self.common, 'live_group_members', return_value=[]), \
             patch.object(self.common.os, 'killpg') as terminate:
            with self.assertRaisesRegex(ValueError, 'operation failed'):
                self.common.monitored(['mock compiler'], self.work, {}, self.work / 'failed')
            terminate.assert_called_once_with(1234, signal.SIGKILL)

    def test_native_group_cleanup_error_still_stops_owned_pg(self):
        owned = {'pid': 42}
        process = Mock(pid=1234, returncode=1)
        process.poll.return_value = 1
        with patch.object(self.pg, 'start', return_value={'identity': owned}), \
             patch.object(self.run.subprocess, 'Popen', return_value=process) as launch, \
             patch.object(self.common, 'exit_status', return_value=1), \
             patch.object(self.common, 'stop_group', side_effect=RuntimeError('group wait failed')), \
             patch.object(self.pg, 'stop') as stop:
            with self.assertRaisesRegex(RuntimeError, 'group wait failed'):
                self.run.native(self.work, {}, 'unused', self.work / 'failed-native')
            self.assertTrue(launch.call_args.kwargs['start_new_session'])
            stop.assert_called_once_with(self.work, {}, owned)

    def test_reaped_leader_pid_reuse_refuses_any_group_signal(self):
        process = Mock(pid=1234)
        with patch.object(self.common.os, 'waitid', side_effect=ChildProcessError('already reaped')), \
             patch.object(self.common.os, 'killpg') as signal_group:
            with self.assertRaises(ChildProcessError):
                self.common.stop_group(process)
            signal_group.assert_not_called()
            process.wait.assert_not_called()

    def test_group_drain_precedes_final_reap(self):
        process = Mock(pid=1234)
        events = []
        def members(group):
            events.append('scan')
            return [1235] if events.count('scan') == 1 else []
        process.wait.side_effect = lambda **kwargs: events.append('reap')
        with patch.object(self.common.os, 'waitid', return_value=Mock(si_status=0, si_code=self.common.os.CLD_EXITED)) as observe, \
             patch.object(self.common.os, 'killpg', side_effect=lambda *args: events.append('signal')), \
             patch.object(self.common, 'live_group_members', side_effect=members), \
             patch.object(self.common.time, 'sleep'):
            self.common.stop_group(process)
        self.assertEqual(events, ['signal', 'scan', 'scan', 'reap'])
        self.assertTrue(observe.call_args.args[2] & self.common.os.WNOWAIT)

    def test_lingering_descendant_fails_boundedly_without_reaping(self):
        process = Mock(pid=1234)
        with patch.object(self.common, 'exit_status', return_value=0), \
             patch.object(self.common.os, 'killpg') as signal_group, \
             patch.object(self.common, 'live_group_members', return_value=[1235]), \
             patch.object(self.common.time, 'monotonic', side_effect=[0, 11]):
            with self.assertRaisesRegex(ValueError, 'did not drain'):
                self.common.stop_group(process)
            signal_group.assert_called_once()
            process.wait.assert_not_called()
            self.assertIn(process, self.common.RETAINED_GROUPS)

    def test_bound_native_build_environment_excludes_ort_override(self):
        module = load('reproduction_build', self.work / 'build.py')
        self.common.CONFIG.update(rustc='mock-rustc', rustdoc='mock-rustdoc')
        with patch.dict(self.common.os.environ, {'ORT_LIB_PATH': '/unrelated', 'ZVEC_AUTO_BUILD': '1'}):
            env = module.build_environment()
        self.assertEqual(env['ZVEC_LIB_DIR'], self.common.CONFIG['native'])
        self.assertEqual(env['ZVEC_AUTO_BUILD'], '0')
        self.assertEqual(env['ORT_LIB_LOCATION'], self.common.CONFIG['ort'])
        self.assertEqual(env['ORT_PREFER_DYNAMIC_LINK'], '1')
        self.assertNotIn('ORT_LIB_PATH', env)

    def test_seed_helper_timeout_stops_owned_pg_and_retains_clone(self):
        from contextlib import nullcontext
        import subprocess
        module = load('reproduction_seed', self.work / 'seed.py')
        install = self.work / 'pg-copy'
        install.mkdir()
        self.common.CONFIG['postgres'] = str(install)
        (self.work / 'build/binaries.json').write_text(json.dumps({'seed': {'path': 'mock', 'sha256': 'pin'}}))
        with patch.object(sys, 'argv', ['seed', '--execute', '--exclusive']), \
             patch.object(self.common, 'exclusive', return_value=nullcontext()), \
             patch.object(self.common, 'source_inputs'), patch.object(self.common, 'assets'), \
             patch.object(self.common, 'digest', return_value='pin'), \
             patch.object(self.common, 'monitored', side_effect=subprocess.TimeoutExpired('mock seed', 900)), \
             patch.object(self.pg, 'stop_seed') as stop:
            with self.assertRaises(subprocess.TimeoutExpired):
                module.main()
            stop.assert_called_once()
            self.assertEqual(stop.call_args.args[0], self.work / 'seed')
        self.assertTrue((self.work / 'seed/.graph-disposable-clone').is_file())
        self.assertFalse((self.work / 'seed/server.json').exists())

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
                            'index_before_bytes': 1000, 'index_after_bytes': 1000,
                            'postgres_disk': {stage: {'logical_bytes': 100, 'allocated_bytes': 4096}
                                              for stage in ['before', 'after']}})
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
            self.assertEqual(result['host_conditions'][0]['cpu_quota'], '400000 100000')
            pg_rows = [r for r in result['disk'] if r['metric'].startswith('owned PG')]
            self.assertEqual(len(pg_rows), 4)
            self.assertTrue(all(r['before'] is not None for r in pg_rows))
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
