"""Tiny runtime identity mocks; no builds, native code, model or PostgreSQL."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import test_templates as fixtures


class RuntimeBindings(unittest.TestCase):
    setUp = fixtures.Templates.setUp
    tearDown = fixtures.Templates.tearDown

    def packet(self):
        return {'postgres_build_identity': {'mapped_libraries': [
            {'location': 'installation/lib/local.so', 'bytes': 2, 'sha256': 'a' * 64},
            {'location': 'external/libfake.so', 'bytes': 3, 'sha256': 'b' * 64}]}}

    def test_common_pg_runtime_accepts_order_and_rejects_each_identity_drift(self):
        first = self.packet(); second = copy.deepcopy(first)
        second['postgres_build_identity']['mapped_libraries'].reverse()
        second['postgres_build_identity']['server_pid'] = 999
        self.assertEqual(self.run.common_postgres_runtime([first, second]),
                         self.run.common_postgres_runtime([first]))
        for field, value in [('location', 'external/other.so'), ('sha256', 'c' * 64), ('bytes', 4)]:
            with self.subTest(field=field):
                changed = copy.deepcopy(first)
                changed['postgres_build_identity']['mapped_libraries'][0][field] = value
                with self.assertRaisesRegex(ValueError, 'cross-job PostgreSQL mapped runtime differs'):
                    self.run.common_postgres_runtime([first, changed])
        changed = copy.deepcopy(first); changed['postgres_build_identity']['mapped_libraries'].pop()
        with self.assertRaisesRegex(ValueError, 'cross-job PostgreSQL mapped runtime differs'):
            self.run.common_postgres_runtime([first, changed])

    def test_resumed_checkpoint_set_rejects_pg_runtime_drift(self):
        out = self.work / 'results'; out.mkdir()
        jobs = self.common.schedule()[:2]
        for job in jobs:
            (out / (job['name'] + '.json')).write_text('{}')
        first = self.packet(); second = copy.deepcopy(first)
        second['postgres_build_identity']['mapped_libraries'][0]['sha256'] = 'c' * 64
        with patch.object(self.run, 'checkpoint', side_effect=[first, second]):
            with self.assertRaisesRegex(ValueError, 'cross-job PostgreSQL mapped runtime differs'):
                self.run.load_checkpoints(out, jobs, 'identity', {}, [], {})
        with patch.object(self.run, 'checkpoint', side_effect=[first, copy.deepcopy(first)]):
            self.assertEqual(len(self.run.load_checkpoints(out, jobs, 'identity', {}, [], {})), 2)

    def mapping_log(self, extra):
        job = self.common.schedule()[0]
        expected = [str((Path(self.common.CONFIG[root]) / name).resolve()) for root, name in
                    [('native', 'libzvec_c_api.so'), ('ort', 'libonnxruntime.so.1.28.0')]]
        bound = {}
        provider_lines = ''
        for role, pin in self.common.MANIFEST['roles'].items():
            path = str(self.work / 'models' / pin['relative'])
            bound[path] = {'sha256': 'synthetic'}
            provider_lines += 'ONNX graph execution-provider assignment model_graph=' + json.dumps(path)
            provider_lines += ' assigned_nodes=' + json.dumps({'CPUExecutionProvider': pin['nodes']}) + '\n'
        encode = lambda path:path.replace('\\', r'\134').replace(' ', r'\040')
        mapped = expected + [expected[0], '/unrelated/libc.so.6'] + extra
        row = {'loaded_libraries': ['1-2 r-xp 0 00:00 1 ' + encode(path) for path in mapped]}
        text = ('test result: ok. 1 passed; 0 failed;\nCACHE_ENGINE_OPEN {"documents":230,"coverage":0.0}\n'
                'CACHE_ENGINE_GRAPH [["mentions",11]]\n' + provider_lines +
                ''.join('CACHE_ENGINE_ROW ' + json.dumps(row) + '\n' for _ in job['sequence'].split(',')))
        return text, job, bound

    def test_extra_ort_zvec_mappings_are_rejected_but_repeated_and_unrelated_are_allowed(self):
        with patch.object(self.common, 'digest', return_value='synthetic'):
            self.run.parse(*self.mapping_log([]))
            extras = ['/other runtime/libonnxruntime.so.1.29.0', '/other/libzvec_c_api.so',
                      '/other/libonnxruntime_providers_shared.so',
                      str((Path(self.common.CONFIG['native']) / 'libzvec_c_api.so').resolve()) + ' (deleted)']
            for extra in extras:
                with self.subTest(extra=extra):
                    with self.assertRaisesRegex(ValueError, 'actual runtime mapping differs'):
                        self.run.parse(*self.mapping_log([extra]))

    def test_runtime_maps_decode_space_tab_newline_and_backslash(self):
        job = self.common.schedule()[0]
        for character in [' ', '\t', '\n', '\\']:
            with self.subTest(character=repr(character)):
                config = dict(self.common.CONFIG)
                config['native'] = str(self.work / ('native' + character + 'dir'))
                config['ort'] = str(self.work / ('ort' + character + 'dir'))
                encode = lambda text: ''.join('\\' + format(ord(char), '03o') if char in ' \t\n\\' else char for char in text)
                paths = [str((Path(config[root]) / name).resolve()) for root, name in
                         [('native', 'libzvec_c_api.so'), ('ort', 'libonnxruntime.so.1.28.0')]]
                raw = [{'loaded_libraries': ['1-2 r-xp 0 00:00 1 ' + encode(path) for path in paths]}
                       for _ in job['sequence'].split(',')]
                text = 'test result: ok. 1 passed; 0 failed;\nCACHE_ENGINE_OPEN {"documents":230,"coverage":0.0}\nCACHE_ENGINE_GRAPH [["mentions",11]]\n'
                text += ''.join('CACHE_ENGINE_ROW ' + json.dumps(row) + '\n' for row in raw)
                bound = {}
                for role, pin in self.common.MANIFEST['roles'].items():
                    path = str(self.work / 'models' / pin['relative'])
                    bound[path] = {'sha256': 'synthetic'}
                    text += 'ONNX graph execution-provider assignment model_graph=' + json.dumps(path)
                    text += ' assigned_nodes=' + json.dumps({'CPUExecutionProvider': pin['nodes']}) + '\n'
                with patch.object(self.common, 'CONFIG', config), patch.object(self.common, 'digest', return_value='synthetic'):
                    self.run.parse(text, job, bound)
                    with self.assertRaisesRegex(ValueError, 'actual runtime mapping differs'):
                        self.run.parse(text.replace('libzvec_c_api.so', 'other.so'), job, bound)


class PublicEntryPoints(unittest.TestCase):
    def test_historical_boundary_is_mixed_without_changing_evidence(self):
        public = fixtures.ROOT.parents[1] / 'results/inference/selected-main-stack-2026-10-01'
        verifier = fixtures.load('public_entrypoint_verifier', public / 'verify.py')
        data = json.loads((public / 'evidence.json').read_text())
        original = copy.deepcopy(data)
        audit = json.loads((public / 'input-scope-audit.json').read_text())
        result = verifier.tables(data, audit)
        self.assertEqual(data, original)
        self.assertEqual(result['conditions']['timed_entrypoint_calls'],
                         {'Engine.search_reranked': 496, 'Engine.search_reranked_with': 384})
        self.assertEqual(result['conditions']['recorded_entrypoint'], 'Engine.search_reranked')
        self.assertEqual(result['conditions']['entrypoint'],
                         'mixed Engine.search_reranked and Engine.search_reranked_with')
        self.assertIn('384 Accurate B-context calls', verifier.markdown(result))
        wrong = copy.deepcopy(data)
        wrong['processes'][0]['calls'][0]['context'] = 'B' if wrong['processes'][0]['calls'][0]['context'] != 'B' else 'A'
        with self.assertRaisesRegex(ValueError, 'mixed timed entrypoint counts differ'):
            verifier.execution_conditions(wrong)
        with self.assertRaisesRegex(ValueError, 'phase/context chronology'):
            verifier.validate(wrong)


if __name__ == '__main__':
    unittest.main()
