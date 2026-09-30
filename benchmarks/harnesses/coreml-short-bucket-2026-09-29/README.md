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
warmup texts and all eight distinct queries per text. It asserts the retained
tokenizer's SHA256, actual CoreML device, one newly scored pair per call,
logical intervals 1–64 / 65–128 / 129–256, and physical padded totals
256 / 512 / 512. [All 24 observed counts](warmup-shape-probe.txt) passed on
the same Apple M4. To repeat it, copy the source to
`crates/pamin-index/tests/scratch_bucket_warmups.rs`, set `PAMIN_EVAL_HOME`
to the cached pinned model home and `PAMIN_DEVICE=auto`, then run
`cargo test -p pamin-index --test scratch_bucket_warmups -- --ignored --nocapture`.
The scratch test is deliberately ignored by the repository; it is a
diagnostic, not a CI-time model download. This later observation proves the
retained tokenizer routes these strings as intended, while the original six
timed arms still lack contemporaneous per-bucket shape logs.
