import json
from pathlib import Path
root = Path(__file__).parent
s = json.loads((root / "summary.json").read_text())
lines = ["# Local backend validation, 2026-09-30", "", "Completed product-entry timing probe: 24 fixed XQuAD queries, three rotated rounds, 432 searches on a shared Apple workstation. Eight disjoint warmup queries; measured reranker calls asserted newly scored == offered > 0. Complete 13,014-document accuracy memory index asserted. These are alternative reranker plans (embedding fixed on CPU), not a new optimization against main.", "", "CoreML ALL allows CPU, GPU and ANE. The GPU arm is CoreML CPUAndGPU. CPU uses ONNX Runtime optimized kernels; its model export differs from CoreML. Accurate stays ALL, Fast stays CPU.", ""]
metrics = [("p50 search ms", "p50_seconds", 1000), ("p95 search ms", "p95_seconds", 1000), ("24-search wall total s", "median_24_search_seconds", 1), ("whole-process wall s (load + warmup + searches)", "median_process_wall_seconds", 1), ("process CPU s (same process scope)", "median_process_cpu_seconds", 1), ("peak process RSS MiB", "median_peak_process_rss_bytes", 1 / 1024**2)]
for tier, reference in [("accurate", "all"), ("fast", "cpu")]:
 lines += ["## " + tier.title(), "", "| Alternative / metric | Current path | Alternative | Absolute delta | Relative delta |", "| --- | ---: | ---: | ---: | ---: |"]
 a = s["arms"][tier + "-" + reference]
 for device in ("all", "gpu", "cpu"):
  if device == reference: continue
  b = s["arms"][tier + "-" + device]
  for title, field, scale in metrics:
   x, y = a[field] * scale, b[field] * scale
   lines.append(f"| {device}: {title} | {x:.3f} | {y:.3f} | {y-x:+.3f} | {100*(y/x-1):+.2f}% |")
  for group in ("same_language", "cross_lingual"):
   x, y = a["ndcg"][group], b["ndcg"][group]
   lines.append(f"| {device}: sample {group} nDCG@10 | {x:.7f} | {y:.7f} | {y-x:+.7f} | {100*(y/x-1):+.2f}% |")
 lines += [""]
lines += ["## Scope and decision", "", "Four prespecified within-tier paired log-time contrasts used 19,999 sign flips (seed 0); Bonferroni family p=0.0002 for all four. This supports the existing per-tier choices on this workload. The 24-query nDCG columns are diagnostic, not acceptance metrics or full benchmark replacements.", "", "All process costs include loading and warmups. Process CPU excludes CoreML services/GPU/ANE work; RSS excludes their allocations. Thus these are neither energy nor model resident memory measurements. Model/index persistent disk deltas and device memory: N/A, not measured. Backend runs did not attest source-weight/tokenizer hashes before and after every process. The compiled test program was frozen; experimental changes are removed after measurement.", "", "## FP64 scorer validation", "", "| Metric | Current FP32 | ORT CPU FP64 | Native SIMD FP64 | Absolute delta (both) | Relative delta |", "| --- | ---: | ---: | ---: | ---: | ---: |", "| 50k recall@10, original f32 oracle | 0.9960 | 0.9960 | 0.9960 | 0 | 0% |", "| 50k recall@10, stored FP16 oracle | 0.9980 | 0.9980 | 0.9980 | 0 | 0% |"]
for group in ("same_language", "cross_lingual"):
 vals = [s["fp64"]["xquad"]["ndcg"][str(m)][group] for m in range(3)]
 assert len(set(vals)) == 1
 lines.append(f"| Full XQuAD {group} nDCG@10 | {vals[0]:.7f} | {vals[1]:.7f} | {vals[2]:.7f} | 0 | 0% |")
vals = [s["fp64"]["miracl"]["ndcg"][str(m)] for m in range(3)]
assert len(set(vals)) == 1
lines += [f"| Full MIRACL-Swahili nDCG@10 | {vals[0]:.7f} | {vals[1]:.7f} | {vals[2]:.7f} | 0 | 0% |", "", "One shared 50k/1024-dimension HNSW memory index, 100 queries and three rotated rounds; independent original-f32 and stored-FP16 cosine oracles. Full XQuAD has 1,190 queries and 13,014 documents; MIRACL has 482 queries and 131,924 documents, both at accuracy profile through search_reranked. Same native ANN candidate pools asserted for every FP64 arm. No returned-list changes; do not adopt an FP64 scorer based on the earlier synthetic near-tie probe. Real-corpus query embedding/reranker scores were shared to isolate vector math, so their call timings cannot compare inference backends. FP64 process CPU, memory and disk deltas: N/A, not measured.", "", "## Device copies and synchronization", "", "Steady ORT profiling (eight warmups removed per node/bucket) places 0.17-0.27% of recorded kernel duration in five CPU integer/mask nodes. Most recorded time is in fused CoreML partitions. CoreML durations include prediction, wrapper handling and waits; they cannot isolate actual GPU/ANE execution or physical transfer cost. Declared intermediate tensor volumes are not measured bus bytes. Keep partition-collapse and mask-broadcast changes as future experiments with product precision and latency gates.", "", "[ORT I/O binding](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html), [pinned CoreML wrapper](https://github.com/microsoft/onnxruntime/blob/v1.28.0/onnxruntime/core/providers/coreml/model/model.mm), [Apple compute plans](https://developer.apple.com/documentation/coreml/mlcomputeplan-85vdw).", "", "summary.json retains the aggregate values. python3 tables.py regenerates this text to stdout without modifying the evidence. Sanitized per-query rows, process resource files, provider attestations, exact experimental patch and run template are retained. python3 verify.py recomputes all backend aggregates and paired tests without modifying evidence. See manifest.json and REPRODUCE.md for identities, order, provenance limits and commands. Device traces are excluded."]
print("\n".join(lines))
