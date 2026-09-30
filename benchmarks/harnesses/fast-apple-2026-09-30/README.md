# Fast Apple backend evidence source

The result is [here](../../results/inference/fast-apple-backend-2026-09-30.md).
This archive retains the exact measurement-only changes to the repository's
`crosslingual` test at experimental commit `bd523771551874fb47f38c70c2cacb6b01bc7b06`.
That intermediate NeuralNetwork commit is a **measurement control, not part
of the final CPU-routing stack**. `inference.patch` reproduces its runtime
change from predecessor `ed3ab210b59f8830182929204693940d70a0a627`.
The product call remains `Engine::search_reranked`; the patches only select
Fast, record every query's ranking and call time, and allow an interrupted arm
to finish a fixed suffix. They never edit a tracked test while a benchmark is
running.

| File | Role |
| --- | --- |
| `coreml.patch` | Derive the complete CoreML Fast measurement from the base test. |
| `inference.patch` | Reproduce the superseded NeuralNetwork runtime control from the stack predecessor. |
| `cpu.patch` | Change only the device premise and label for the CPU arm. |
| `resume.patch` | Run the fixed 760–1,189 suffix with a 10-query overlap. |
| `Cargo.toml.in`, `Cargo.lock` | Build each copy as an independent release test binary. |
| `run-fast-current.py.in` | Run CPU then CoreML sequentially, checking model/source/binary and memory/disk guards. |
| `run-fast-resume.py.in` | Finish the interrupted CoreML arm and reject unequal overlap rankings. |
| `sources.json` | Exact source, model, binary, host, and provider identities for this run. |
| `verify_rows.py` | Recheck all 1,190 query IDs and the ten exact overlap rankings in the archived JSON. |

To regenerate the three harness sources, keep this archive in one checkout
(`FAST_ARCHIVE_DIR`) and check out the experimental commit in a separate
isolated source checkout (`EVIDENCE_REPO`). Then run:

```sh
python3 - <<'PY'
import os, pathlib, subprocess
repo = pathlib.Path(os.environ['EVIDENCE_REPO']).resolve()
archive = pathlib.Path(os.environ['FAST_ARCHIVE_DIR']).resolve()
out = pathlib.Path(os.environ['FAST_EVIDENCE_HOME']).resolve()
base = subprocess.check_output([
    'git', '-C', str(repo), 'show',
    'bd523771551874fb47f38c70c2cacb6b01bc7b06:crates/pamin-engine/tests/crosslingual.rs',
])
out.mkdir(parents=True, exist_ok=True)
for arm in ('coreml', 'cpu', 'resume'):
    directory = out / arm
    directory.mkdir(exist_ok=True)
    (directory / 'lib.rs').write_bytes(base)
    patches = ('coreml.patch',) if arm == 'coreml' else ('coreml.patch', f'{arm}.patch')
    for name in patches:
        patch = (archive / name).read_text().replace('${EVIDENCE_REPO}', str(repo))
        subprocess.run(['patch', '-s', 'lib.rs'], cwd=directory,
                       input=patch, text=True, check=True)
    manifest = (archive / 'Cargo.toml.in').read_text().replace('${EVIDENCE_REPO}', str(repo))
    if arm != 'coreml':
        manifest = manifest.replace('pamin-fast-coreml-program', f'pamin-fast-{arm}')
    (directory / 'Cargo.toml').write_text(manifest)
    (directory / 'Cargo.lock').write_bytes((archive / 'Cargo.lock').read_bytes())
PY
```

The patch archive was independently applied and the regenerated source bytes
were checked against all three frozen source SHA256 values in `sources.json`.
If the intermediate commit is unavailable, start from `ed3ab21` in an
isolated source checkout and apply `inference.patch` there to reproduce the
runtime tree. A new local commit will have a different OID; replace the
runner's HEAD premise only after checking its tree equals the archived
control. Never merge that experimental format patch as a separate product
change.
`python3 benchmarks/harnesses/fast-apple-2026-09-30/verify_rows.py` independently
checks the archived raw-row merge and overlap.
Build each test with the pinned toolchain and a shared release target, then
copy the produced test executable out of Cargo's target directory before
running another build. The exact original run's frozen binary hashes are in
`sources.json`; a fresh build may have different binary bytes, so update the
runner's binary-hash premise to the fresh frozen copy before replaying. Do
not disable its model/source, query-count, execution-provider, or resource
checks. `PAMIN_EVAL_HOME` must hold the complete revision-bound XQuAD-R index,
and the Fast model files must match
[`xquad-r-fast-model-artifacts.json`](../../results/retrieval/xquad-r-fast-model-artifacts.json).
The [base product-search archive](../product-search-2026-09-30/README.md)
has the dataset preparation and `score.rs`/`compare.rs` owners. Instantiate
the two `.py.in` runners by replacing their named `${...}` paths with the
actual checkout, evaluation home, evidence directory, ORT library directory,
scorer binary and judgments file. Run sequentially, without a concurrent
model build. The first CoreML process here was rejected when free disk fell
below 10 GiB; `resume.patch` and the second runner recovered **quality** only
after verifying all ten overlap rankings and both scores byte for byte.

The stitched CoreML latency distribution describes these two real product
processes; it is not a clean single-process timing arm. CPU time excludes
CoreML services and GPU/ANE work, and EP node assignment does not reveal
CoreML's internal device placement. Keep those boundaries when reusing the
numbers.
