"""Verify retained evidence read-only and regenerate summaries in scratch space."""
import contextlib
import gzip
import hashlib
import io
import json
import shutil
import tempfile
from pathlib import Path
from tables import allocation_table, runtime_table

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'manifest.json').read_text())
for name, identity in manifest['artifacts'].items():
    data = (root / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == identity['public_sha256'], name
    assert not any(prefix in data for prefix in (b'/Users/', b'/private/tmp/', b'.codex/worktrees'))
raw_input = gzip.decompress((root / 'xquad-fixed-candidates.json.gz').read_bytes())
assert hashlib.sha256(raw_input).hexdigest() == manifest['input_sha256']
inputs = json.loads(raw_input)
assert len(inputs) == 1190
expected_ids = [inputs[index]['id'] for index in manifest['sample_indices']]
for arm in ('baseline', 'candidate'):
    for round_index in range(3):
        rows = json.loads((root / f'{arm}-{round_index}.json').read_text())
        assert [row['id'] for row in rows] == expected_ids
        assert all(len(row['scores']) == 30 for row in rows)
counts = json.loads((root / 'allocations.json').read_text())
assert len(counts) == 4 and all(row['after_calls'] == 2 for row in counts)
expected = json.loads((root / 'summary.json').read_text())
with tempfile.TemporaryDirectory() as temporary:
    scratch = Path(temporary)
    for arm in ('baseline', 'candidate'):
        for round_index in range(3):
            for extension in ('json', 'resource'):
                name = f'{arm}-{round_index}.{extension}'
                shutil.copyfile(root / name, scratch / name)
    source = (root / 'sources/summarize.py.in').read_text().replace('${RUN_ROOT}', str(scratch))
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(source, 'retained_summary', 'exec'), {})
    assert json.loads((scratch / 'summary.json').read_text()) == expected
assert expected['same_scores_across_all_arms']
readme = (root / 'README.md').read_text()
assert allocation_table(counts) in readme
assert runtime_table(expected) in readme
print('Verified exact paired scores, allocation deltas and process aggregates without modifying the archive.')
