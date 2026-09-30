import gzip, json, statistics
from collections import defaultdict

def compute_profiling(root):
    result = {}
    for shape in ("4x128", "4x64", "2x256"):
        events = json.loads(gzip.decompress((root / "profiling" / (shape + ".json.gz")).read_bytes()))
        groups = defaultdict(list)
        for event in events:
            assert event["duration_us"] >= 0
            groups[(event["node"], event["provider"])].append(event["duration_us"])
        assert len(groups) == 7
        nodes = []
        for (name, provider), times in groups.items():
            assert len(times) > 8
            steady = times[8:]
            nodes.append({"node": name, "provider": provider, "calls": len(times), "warmups_removed": 8, "steady_total_us": sum(steady), "steady_median_us": statistics.median(steady)})
        total = sum(n["steady_total_us"] for n in nodes)
        cpu = sum(n["steady_total_us"] for n in nodes if n["provider"] == "CPUExecutionProvider")
        result[shape] = {"nodes": nodes, "steady_kernel_us": total, "cpu_kernel_us": cpu, "cpu_fraction": cpu / total}
    return result
