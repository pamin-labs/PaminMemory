# Retained short-bucket resource collectors

These are the actual observer, controller/resume and summarizer used for the six
completed arms. Named placeholders replace machine-specific paths; hashes retain
the original and template identities. Replace RUN_ROOT/CANDIDATE_REPO/CONTROL_REPO
before running the Python scripts. The ranking harness, its invocation arguments,
control patch, binary hashes and complete per-query timings remain in the linked
results JSON. Read the controller and observer together: the observer inventories
logical temporary package bytes, and the controller samples process RSS; neither
measures unique APFS allocation or all CoreML service/device memory.

The original physical-token warmup assertion was insufficient to distinguish
128/256 shapes. The evidence JSON preserves it unchanged and supplies a corrected
harness with logical-token intervals for new runs. Existing timing observations
retain that limitation; a newly validated rerun is separate evidence.

The [later real-model probe](warmup-shape-probe.rs) uses the original three
warmup texts and all eight distinct queries per text. It rejects an alternate
Hugging Face cache or endpoint, checks that the loader's `refs/main` selects
the pinned revision before and after loading, and hashes that revision's
tokenizer. It also asserts the loaded 256-token limit, actual CoreML device,
nonzero CoreML graph assignment in each of the three static scoring sessions,
one newly scored pair per call, logical intervals 1–64 / 65–128 / 129–256,
and physical padded totals 256 / 512 / 512. The [new real-model run](warmup-shape-probe.txt)
found 765 CoreML and 5 CPU assigned nodes in each session; all 24 warmup
counts passed on the same Apple M4. Node assignment does not reveal CoreML's
internal CPU/GPU/ANE placement. To repeat it from the repository root:

```sh
cp benchmarks/harnesses/coreml-short-bucket-2026-09-29/warmup-shape-probe.rs crates/pamin-index/tests/scratch_bucket_warmups.rs
env -u HF_HOME -u HF_ENDPOINT PAMIN_EVAL_HOME=/path/to/cached-eval-home PAMIN_DEVICE=auto cargo test -p pamin-index --test scratch_bucket_warmups -- --ignored --nocapture
rm crates/pamin-index/tests/scratch_bucket_warmups.rs
```

The model cache in `/path/to/cached-eval-home/models` must contain the pinned revision.
The scratch test is deliberately ignored by the repository; it is a
diagnostic, not a CI-time model download. This later observation proves the
retained tokenizer routes these strings as intended, while the original six
timed arms still lack contemporaneous per-bucket shape logs.
