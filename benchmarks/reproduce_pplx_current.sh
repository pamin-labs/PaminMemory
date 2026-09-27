#!/usr/bin/env bash
set -euo pipefail

: "${PAMIN_EVAL_HOME:?set PAMIN_EVAL_HOME to a scratch workspace with the XQuAD-R files}"
cd "$(git rev-parse --show-toplevel)"
scratch=crates/pamin-engine/tests/scratch_pplx_greek.rs
rm -f "$scratch"
trap 'rm -f "$scratch"' EXIT

# The model, index and temporary files need headroom beyond the persisted
# workspace. Keep at least 10 GiB free after a roughly 2 GiB trial peak.
available_kib=$(df -Pk "$PAMIN_EVAL_HOME" | awk 'NR == 2 {print $4}')
if (( available_kib < 12 * 1024 * 1024 )); then
  echo "need at least 12 GiB available before the pplx trial" >&2
  exit 1
fi

PAMIN_PROFILE=pplx cargo test -p pamin-engine --test crosslingual \
  search_reaches_across_languages -- --ignored --nocapture \
  > "$PAMIN_EVAL_HOME/pplx-current-full.log" 2>&1

python3 benchmarks/pplx_prepare_xquad.py --eval-home "$PAMIN_EVAL_HOME"
cp benchmarks/pplx_greek.rs "$scratch"
PAMIN_PROFILE=pplx cargo test -p pamin-engine --test scratch_pplx_greek \
  -- --ignored --nocapture > "$PAMIN_EVAL_HOME/pplx-current-greek.log" 2>&1
python3 benchmarks/pplx_greek_stats.py \
  "$PAMIN_EVAL_HOME/pplx-greek-paired.json" \
  > "$PAMIN_EVAL_HOME/pplx-greek-summary.txt"

echo "raw and summary results are in $PAMIN_EVAL_HOME"
