# Current paired lexical-channel sweep evidence

This is a **new** complete run at public code commit
`f97635099e08aa8696aa3b0237424d486a255ecd`, not a reconstruction of
missing 2026-09-27 per-query matrices. It runs `CHANNELS=1` before reranking
through the real `Engine::search_fused` trace, then replays the project's
`Fusion::fuse` over the same candidates. `CHANNELS_OUT` now persists the exact
`Scores` vectors and all 37 settings from XQuAD-R's dedicated diagnostic;
MuSiQue already used the shared `Diagnosis` output owner. Complete stdout is
retained for every printed grid row, fold choice, p value and channel table.

| Corpus | Questions | Shipped macro nDCG@10 | Raw paired file | Full table and fold output |
| --- | ---: | ---: | --- | --- |
| XQuAD-R, 13,014 indexed sentences | 1,190, two answer groups | 0.7405 | [JSON](xquad-paired.json) | [stdout](xquad-stdout.log) |
| MuSiQue, 10,785 memories | 1,000, two-hop group | 0.6814 | [JSON](musique-paired.json) | [stdout](musique-stdout.log) |

XQuAD-R used a complete `xquad-accuracy-24ad7f1862182925-705ffaf38623fb5c`
named-passage memory index and the pinned embedding identity
`705ffaf38623fb5c71a2ced99d8674e45fb19e8ad05f263bfa09474e4d9b153f`.
Its JSON records the code commit, project, identity, all per-query scores,
per-language results and 37 settings. MuSiQue reused the cached 10,785-memory
project with 12,840 live edges and the first 1,000 questions. Its current
index marker does not attest the exact model-source revision; the raw grid is
auditable, but it cannot retroactively establish that missing provenance.
Both runs used the `accuracy` profile and in-memory vector index on the shared
Apple M4. They are fusion-only diagnostics, not product reranker latency or
a new default weight decision.

The current XQuAD-R lexical-only held-out procedure scored 0.7419 versus the
shipped 0.7405 (`p=0.2321`); MuSiQue scored 0.6829 versus 0.6814
(`p=0.3811`). Folds do not choose one stable pair across corpora, so no new
weight is selected. The historical table and complete printed output remain
[separate](../lexical-2026-09-27/README.md); their unrecorded matrices cannot
be recreated from this run. The first 2026-09-30 XQuAD archive omitted an
empty `lexical_segmented` ranking for one query and had no `graph` standalone
vector. Its 1,189-query standalone mean of 0.740896 for same-language is now
0.740274 over all 1,190 queries; cross-language is 0.026661 → 0.026638.
The shipped fused macro nDCG@10 stays 0.7405, and MuSiQue's paired output is
byte-identical after the same code fix. Run
`python3 benchmarks/results/fusion/lexical-2026-09-30/verify.py` to check that
every channel and variant has one score per query and that the printed shipped
means agree. The scoring loops preserve query order.

To rerun with prepopulated revision-bound XQuAD-R and cached MuSiQue projects,
set `PAMIN_EVAL_HOME`, `PAMIN_DEVICE=cpu`, `PAMIN_PROFILE=accuracy`,
`PAMIN_VECTOR_INDEX=memory`, and `CHANNELS=1`; set `CHANNELS_OUT` to a new JSON
path per corpus. XQuAD-R runs the ignored `crosslingual` test
`search_reaches_across_languages`; MuSiQue runs the ignored `multihop` test
`search_answers_questions_that_take_several_steps` with
`MUSIQUE_QUESTIONS=1000` and its cached `MUSIQUE_DIR`. Keep each test's full
stdout next to its JSON. A run that does not write the expected complete raw
rows must not be published as a sweep result.

The committed JSON is whitespace-packed for review; parsed content was checked
equal to the harness's corrected pretty-printed files. Their original SHA256s
were `26fa6ce967ee90dc5757d52e187ed69c18736d6359f43a78b6dcac185f8974b1`
(XQuAD-R) and `a18e913fef82afe93a6c686c224578b778c777a524a9813f78a9e3947185b2ad`
(MuSiQue).
