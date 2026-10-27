# ConditionalQA: sections, `part_of` and parent context, 2026-10-26

The harness behind
[`results/ingest/conditionalqa-sections-2026-10-26.md`](../../results/ingest/conditionalqa-sections-2026-10-26.md).
It produces numbers and asserts its own premises only, so it is kept as blocks
appended to `crates/pamin-engine/tests/sectioning.rs`, following `pamin-dev`.

- `outline.rs.append` -- shared helpers (`view_of`, `file_part`), and the
  Qasper and repository-Markdown outline comparison they were first written
  for.
- `cqa.rs.append` -- `scratch_cqa`: the corpus and three passes. Whole pages,
  heading sections, and the same sections once `part_of` and next-sibling
  `related_to` edges are stated. It needs the outline API of `920af63`.
- `cqa_partof.rs.append` -- `scratch_cqa_partof`: the sections project of a
  `scratch_cqa` run (`CQA_NONCE`), restated with `part_of` alone, searched
  again.
- `parents.rs.append` -- `scratch_cqa_parents`: one pass over that project
  whose hits carry their parent section, shown three ways. It was run on a
  build whose `SearchHit::parent` held the parent's name and opening. What
  ships keeps the name only, so this arm needs that build to re-run.
- `spread.py CQA_DIR QASPER_DIR` -- how each question's evidence is spread
  over its document's sections, for ConditionalQA and for Qasper.
- `cqa_summarise.py rows.json base:new ...` and `parents_summarise.py` -- the
  paired tables, by spread.

Each pass asserts what it measures:

- the project holds exactly the spans `Engine::cut` predicts;
- every edge stated is live, and none left over;
- every question returns hits;
- every evidence sentence is in its page;
- a parent is another section of the hit's own file.

## Commands

```sh
git clone --depth 1 https://github.com/haitian-sun/ConditionalQA /data/ConditionalQA
export PAMIN_EVAL_HOME=/path/to/eval CQA_DIR=/data/ConditionalQA/v1_0 CQA_TRAIN=600
H=benchmarks/harnesses/conditionalqa-sections-2026-10-26
T=crates/pamin-engine/tests/scratch_cqa.rs

rm -f crates/*/tests/scratch_*.rs
cat crates/pamin-engine/tests/sectioning.rs $H/outline.rs.append $H/cqa.rs.append \
  $H/cqa_partof.rs.append > $T
SCRATCH_OUT=cqa.json cargo test --release -p pamin-engine --test scratch_cqa -- \
  --ignored --exact scratch_cqa --nocapture
# The project name it printed ends in the run's nonce.
CQA_NONCE=<nonce> SCRATCH_OUT=cqa-partof.json cargo test --release -p pamin-engine \
  --test scratch_cqa -- --ignored --exact scratch_cqa_partof --nocapture
rm -f crates/*/tests/scratch_*.rs

python3 -c "import json; json.dump(json.load(open('cqa.json')) + json.load(open('cqa-partof.json')), open('all.json', 'w'))"
python3 $H/cqa_summarise.py all.json whole:sections sections:partof outline:partof
```

As with the Qasper harness, the container these ran in supplied PostgreSQL
17.6 through `PAMIN_EVAL_POSTGRES_DIR` and ran as an unprivileged user.
