The prospective `run-guarded.py.in` adds per-copy evidence without changing the archived `run.py.in` trial loop. These guards were not used historically and cannot recover historical per-copy identity or rule out mutations during the nine archived trials.

After preparation has stopped, create a complete expected fixture manifest in a **separate preparation invocation**. Keep it outside the source seed and future result directory, with mode0600. It includes credential-file hashes and complete relative paths: retain it privately, and publish only sanitized aggregate identity/counts. The runner requires this supplied expected manifest; it never derives its own expected contents from the trial's mutable source.

Copy the wrapper and unchanged `run.py.in` beside each other. In that directory the following command creates a private manifest from an explicitly selected stopped seed and pinned shared model directory; `NEW_MANIFEST` must not exist:

```sh
python3 - SEED MODEL_CACHE NEW_MANIFEST <<'PY'
import importlib.machinery,importlib.util,json,os,sys
from datetime import datetime,timezone
from pathlib import Path
sys.dont_write_bytecode=True
loader=importlib.machinery.SourceFileLoader('guards','run-guarded.py.in')
spec=importlib.util.spec_from_loader(loader.name,loader)
guards=importlib.util.module_from_spec(spec);loader.exec_module(guards)
seed,models,output=map(Path,sys.argv[1:])
assert not output.resolve().is_relative_to(seed.resolve())
fixture=guards.fixture_snapshot(seed,models)
record={'recorded_utc':datetime.now(timezone.utc).isoformat(),'fixture':fixture}
descriptor=os.open(output,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(descriptor,'w') as stream:json.dump(record,stream,indent=2);stream.write('\n')
PY
```

Add `--fixture-manifest NEW_MANIFEST` to the `trials` subcommand of the guarded runner, alongside its existing seed/output/arm arguments. Keep `--record NEW_PRIVATE_RECORD` outside the seed and result directory. Both setup and trials additionally require the separately prepared `--model-manifest` described in [model-guards-README.md](model-guards-README.md); setup does not require a stopped fixture manifest. Both modes reject inherited `HF_HOME` and `HF_HUB_CACHE` before creating output, as well as the existing inherited `PAMIN_*` tuning guards.

For each exact root seed-to-trial copy, the wrapper hashes the complete stopped source against that expected manifest, delegates to the original `shutil.copytree`, hashes the complete copied contents/modes/symlink targets, rejects shared regular-file inodes and unexpected external symlinks, rechecks the source, and verifies the arm's attested executable SHA/bytes before returning control to the original runner. The sole external symlink exception is the exact pinned `models` directory; its contents require the separate model/weight identity checks and are not scanned by this fixture manifest. Recursive subtree copies are delegated without duplicate attestations. All copy/hash checks finish before the unchanged runner starts that process's elapsed timer.

The mode0600 private configuration record retains nine ordered timestamped source/clone aggregate tree digests, file/entry counts, pinned model target and arm/commit/binary identities. It contains no per-file credential/path mapping. Model evidence contains aggregate selected-model/asset/released-source counts; the full model role map stays in its separate private manifest. Completion requires exactly all nine rotated copies and an unchanged full source snapshot afterward. Late failures preserve the partial record and timestamp. Scoped `copytree`, environment and `sys.argv` changes are restored in `finally`; trial logic remains in the immutable historical runner.

Run the inert `test-fixture-guards.py.in` with ordinary Python for tiny fake fixtures only. It covers nested recursion, source/clone mutation, shared inodes, external links, changed executables, eight/ten-copy rejection, post-source mutation, late failure, permissions and process-state restoration. It launches no PostgreSQL, model, compiler or genuine trial. Endpoint hashes and pre-launch checks do not certify every intervening instant; retain that limit when interpreting evidence.
