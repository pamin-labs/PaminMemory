"""Verify retained paired input-padding evidence and regenerate its summary."""
import contextlib
import gzip
import hashlib
import io
import json
from pathlib import Path

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
source = (root / 'sources/summarize.py.in').read_text().replace('${RUN_ROOT}', str(root))
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(source, 'retained_summary', 'exec'), {})
assert json.loads((root / 'summary.json').read_text()) == expected
assert expected['same_scores_across_all_arms']
print('Verified 60 queries × 30 candidates × 3 rounds per arm; exact scores and complete allocation evidence.')
