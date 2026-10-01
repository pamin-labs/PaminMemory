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

Add `--fixture-manifest NEW_MANIFEST` to the `trials` subcommand of the guarded runner, alongside its existing seed/output/arm arguments. Keep `--record NEW_PRIVATE_RECORD` outside the seed and result directory. Both setup and trials additionally require the separately prepared `--model-manifest` described in [model-guards-README.md](model-guards-README.md); setup does not require a stopped fixture manifest. Both modes reject inherited `HF_HOME` and `HF_HUB_CACHE` before creating output, as well as the existing inherited `PAMIN_*` tuning guards. They also reject inherited `TOKIO_WORKER_THREADS` (including an empty value), then set and record `TOKIO_WORKER_THREADS=4` for the retained multi-thread Tokio helper. This prospective value is separate from `PAMIN_INFERENCE_THREADS=4`; it cannot recover historical Tokio concurrency.

For each exact root seed-to-trial copy, the wrapper hashes the complete stopped source against that expected manifest, delegates to the original `shutil.copytree`, hashes the complete copied contents/modes/symlink targets, rejects shared regular-file inodes and unexpected external symlinks, rechecks the source, and verifies the arm's attested executable SHA/bytes before returning control to the original runner. The sole external symlink exception is the exact pinned `models` directory; its contents require the separate model/weight identity checks and are not scanned by this fixture manifest. Recursive subtree copies are delegated without duplicate attestations. All copy/hash checks finish before the unchanged runner starts that process's elapsed timer.

The mode0600 private configuration record retains nine ordered timestamped source/clone aggregate tree digests, file/entry counts, pinned model target and arm/commit/binary identities. It contains no per-file credential/path mapping. Model evidence contains aggregate selected-model/asset/released-source counts; the full model role map stays in its separate private manifest. Completion requires exactly all nine rotated copies and an unchanged full source snapshot afterward. Late failures preserve the partial record and timestamp. Scoped `copytree`, environment and `sys.argv` changes are restored in `finally`; trial logic remains in the immutable historical runner.

Run the inert `test-fixture-guards.py.in` with ordinary Python for tiny fake fixtures only. It covers nested recursion, source/clone mutation, shared inodes, external links, changed executables, eight/ten-copy rejection, post-source mutation, late failure, permissions and process-state restoration. It launches no PostgreSQL, model, compiler or genuine trial. Endpoint hashes and pre-launch checks do not certify every intervening instant; retain that limit when interpreting evidence.

Guarded setup/trials also reject inherited `TOKENIZERS_PARALLELISM` and `RAYON_NUM_THREADS`, including empty values, then explicitly set and record `TOKENIZERS_PARALLELISM=false` and `RAYON_NUM_THREADS=4`. The serial tokenizer setting and fixed Rayon pool are prospective controls; historical tokenizer/Rayon concurrency remains unknown.

The prospective runner rejects every inherited `MALLOC_*` variable and
`GLIBC_TUNABLES` (which can also configure glibc allocation), including empty
values, before setup or trial work. It uses the default allocator environment.
Historical allocator tuning was not retained and remains unknown; this guard
does not certify the historical memory or timing conditions.

Every resolved attested arm executable must be outside setup home and, for
trials, outside both seed and output roots. This includes paths through symlink
ancestors. Validation runs before writing records, copying fixtures or launching
setup/trials, so build artifacts cannot enter measured index-disk accounting.

Prospective bootstrap and seed setup require the attested `candidate` arm and
its candidate commit. Conversion logs must resolve outside the entire measured
stopped seed, including paths through symlink ancestors, before any log is opened.
These controls do not add missing historical setup attestations.

Prospective setup and every trial now copy each attested executable into an external mode-0700 temporary directory, set copies to mode 0500, open them read-only, and invoke `/proc/self/fd/<fd>` with Python `pass_fds`. Byte count, SHA-256, device and inode endpoints identify the opened file actually supplied to each direct invocation even if a pathname is replaced. The private setup/trial record retains the endpoints and actual FD command; FD numbers themselves are local, ephemeral labels. Copies and descriptors are released on success and failure. The historical `run.py.in` and accepted archives remain byte-identical. The prospective wrapper's trial elapsed interval includes executable endpoint hashing/recording; its timings are not interchangeable with the historical unwrapped interval. These checks do not certify every intervening instant or exclude same-user writes to an opened inode between endpoints.

Conversion attestations, original executables, logs and private executable copies must resolve outside the fixed measured Disk seed before reads or launch. `test-execution-binding.py.in` uses only tiny fake bytes and mocked invocations, including path replacement, in-place mutation, failure cleanup and conversion seed-boundary rejection.

Version2 prospective fixture manifests additionally bind every regular file and symlink under
`index/` to its `st_blocks * 512` allocation. Source and clone endpoint checks
require the same per-entry allocation as the separately prepared manifest, and
private copy receipts retain the aggregate starting index allocation. Copies
that preserve bytes but change allocation are rejected before launch; sparse,
CoW or compressed copies may therefore require a different controlled copy
method, rather than silently passing. This certifies the checked starting
allocation endpoints, not physical exclusive storage or continuous allocation.
Historical per-copy starting allocation was not recorded and remains N/A;
the retained allocated-byte observations cannot exclude starting-copy effects.
Version1 manifests must be prepared again for prospective runs. The attestation
is read once: the parsed document and recorded SHA-256 use the same byte buffer,
so a concurrent pathname replacement cannot identify a different document.
