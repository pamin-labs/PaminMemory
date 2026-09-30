# CoreML prepare cache-hit replay

The [result and limits](../../results/inference/coreml-stream-cache-recheck-2026-09-30.md)
separate two 30-round runs. `strict-hit-rerun.py` is the exact source of the
second, stricter run; its predecessor made the same `source.onnx`/`model.onnx`,
graph-hash and unchanged-inode checks but did not rehash each frozen executable
or check free memory before every arm. Both raw row sets are archived.

The script consumes the frozen binary paths, model source and cache paths in
the [original protocol](../../results/inference/coreml-stream-cache-2026-09-29.json).
Those paths describe this machine; on another machine, reproduce the two
release binaries from that protocol, point the protocol at the corresponding
pinned files, and keep its hashes. Run from the repository root with the
runtime library directory in `DYLD_LIBRARY_PATH`. The script alternates order
for 30 rounds, exits if either cache file is missing or replaced, and saves a
row after every successful arm. It does not touch tracked source files.

To check the committed rows without models or a runtime library:

```sh
python3 benchmarks/harnesses/stream-cache-recheck-2026-09-30/summarise.py
```

The summariser uses one fixed random seed and 100,000 sign flips for each
run's within-round wall-time differences. It reports the two runs separately;
combining them would hide the host-load change that invalidated a stable speed
claim. Process RSS is sampled by `wait4` and excludes system services.
