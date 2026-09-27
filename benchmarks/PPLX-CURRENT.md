# Reproducing the rejected pplx embedding arm

This **experimental branch is not intended for merge**. It restores only the
optional `pplx` profile needed to re-run the 2026-09-27 comparison under the
current full-head reranker. The default remains BGE-M3. The official MIT model
repository is pinned to revision `2c4d510dd4a732063c31a0f70193e35067b51fd8`;
its int8 `pooler_output_int8` is normalized per text, with one text per ONNX
pass. Batching changed a vector on real XQuAD-R text and is excluded.

Set `PAMIN_EVAL_HOME` to a scratch workspace that already has the 11 public
XQuAD-R JSON files and the named BGE-M3 accuracy index for fingerprint
`24ad7f1862182925`. The normal crosslingual test can download the corpus and
build an index, but the paired Greek arm refuses an empty index. Keep at least
12 GiB free before starting; the script checks this. Model downloads and the
first pplx index build can take hours on a busy CPU. Use a separate workspace
for any new run, and do not read single-run wall time as a model speedup.

```sh
PAMIN_EVAL_HOME=/path/to/scratch-workspace bash benchmarks/reproduce_pplx_current.sh
```

The script writes raw logs and the Greek per-question JSON under the workspace,
then removes its ignored `scratch_pplx_greek.rs` test target. It does not edit
tracked files. `pplx-current-summary.json` is the archived full-corpus summary;
`pplx-greek-paired-2026-09-27.json` is the archived 108-question score matrix.
The scorer uses fixed-seed bootstrap and sign-flip tests. The aggregate BGE
column comes from a separate run on the same corpus and corrected key, so it
has no paired p-value; the Greek result is paired through the same product
entry point.

The repository's three-run median rule applies before publishing latency.
This reproduction recipe records accuracy and the failed precision gate, not
a timing comparison.
