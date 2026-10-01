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


if __name__ == '__main__':
    unittest.main()
