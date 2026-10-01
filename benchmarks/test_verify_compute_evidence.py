"""Small, model-free checks for retained cost evidence and resealed counterexamples."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import verify_compute_evidence as verifier


class CostEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.queries = [r for r in verifier.rows("xquad-bge.jsonl") if r["kind"] == "query"]
        cls.costs = json.loads((verifier.ROOT / "cost-summary.json").read_text())
        cls.raw = {f"{name}-{block['block']}.jsonl": verifier.rows(f"{name}-{block['block']}.jsonl")
                   for name, arm in cls.costs.items() for block in arm["blocks"]}

    def check(self, raw, costs):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cost-summary.json").write_text(json.dumps(costs))
            with patch.object(verifier, "ROOT", root), patch.object(verifier, "rows", side_effect=raw.__getitem__):
                verifier.verify_costs(self.queries)

    def test_retained_costs(self):
        self.check(self.raw, self.costs)

    def test_each_block_query_identity_and_order(self):
        for filename in self.raw:
            for change in ("id", "language", "order"):
                with self.subTest(filename=filename, change=change):
                    raw = dict(self.raw)
                    altered = copy.deepcopy(raw[filename])
                    queries = [r for r in altered if r["kind"] == "query"]
                    if change == "order":
                        queries[0]["id"], queries[1]["id"] = queries[1]["id"], queries[0]["id"]
                        queries[0]["language"], queries[1]["language"] = queries[1]["language"], queries[0]["language"]
                    else:
                        queries[0][change] = "different"
                    raw[filename] = altered
                    with self.assertRaisesRegex(AssertionError, "cost query slice/order changed"):
                        self.check(raw, self.costs)

    def test_consistent_wrong_slice_still_rejected(self):
        raw = copy.deepcopy(self.raw)
        for block in raw.values():
            for row in block:
                if row["kind"] == "query":
                    row["id"] = "other-" + row["id"]
        with self.assertRaisesRegex(AssertionError, "cost query slice/order changed"):
            self.check(raw, self.costs)

    def test_resealed_wrong_arm_manifests(self):
        changes = {"code": "wrong-source", "profile": "speed", "device_policy": "other",
                   "depth": 10, "limit": 10, "index": "other", "documents": 1,
                   "queries": 65, "compiled_cache_warmth": "cold", "shared_host": False}
        for name in self.costs:
            for field, value in changes.items():
                with self.subTest(arm=name, field=field):
                    raw = dict(self.raw)
                    filename = f"{name}-0.jsonl"
                    raw[filename] = copy.deepcopy(raw[filename])
                    raw[filename][0][field] = value
                    costs = copy.deepcopy(self.costs)
                    costs[name]["blocks"][0]["manifest"][field] = value
                    with self.assertRaisesRegex(AssertionError, f"unexpected {field}"):
                        self.check(raw, costs)

    def test_main_control_ranking_parity_each_block(self):
        for block in range(3):
            raw = dict(self.raw)
            filename = f"main-cpu-{block}.jsonl"
            raw[filename] = copy.deepcopy(raw[filename])
            query = next(r for r in raw[filename] if r["kind"] == "query")
            query["ranked"] = ["changed-ranking"]
            with self.subTest(block=block), self.assertRaisesRegex(AssertionError, "main CPU ranking differs"):
                self.check(raw, self.costs)

    def test_each_published_manifest_is_checked(self):
        for name, arm in self.costs.items():
            for block in arm["blocks"]:
                costs = copy.deepcopy(self.costs)
                next(b for b in costs[name]["blocks"] if b["block"] == block["block"])["manifest"]["open_seconds"] += 1
                with self.subTest(arm=name, block=block["block"]), self.assertRaisesRegex(AssertionError, "raw and summary manifests differ"):
                    self.check(self.raw, costs)

    def test_raw_summary_manifest_parity_includes_measurements(self):
        raw = dict(self.raw)
        raw["main-cpu-0.jsonl"] = copy.deepcopy(raw["main-cpu-0.jsonl"])
        raw["main-cpu-0.jsonl"][0]["open_seconds"] += 1
        with self.assertRaisesRegex(AssertionError, "raw and summary manifests differ"):
            self.check(raw, self.costs)


class DeviceProofTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proof = json.loads((verifier.ROOT / "device-and-cache-proof.json").read_text())
        cls.artifacts = {p["event_artifact"]: json.loads((verifier.ROOT / "logs" / p["event_artifact"]).read_text()) for p in cls.proof}

    def check(self, proof, artifacts):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "logs").mkdir()
            proof = copy.deepcopy(proof)
            for entry in proof:
                data = json.dumps(artifacts[entry["event_artifact"]]).encode()
                (root / "logs" / entry["event_artifact"]).write_bytes(data)
                entry["event_sha256"] = hashlib.sha256(data).hexdigest()
            (root / "device-and-cache-proof.json").write_text(json.dumps(proof))
            with patch.object(verifier, "ROOT", root):
                verifier.verify_device_proof()

    def test_retained_device_proof(self):
        self.check(self.proof, self.artifacts)

    def test_all_arms_cannot_reuse_the_cpu_control_artifact(self):
        proof = copy.deepcopy(self.proof)
        control = proof[0]
        for entry in proof[1:]:
            entry.update({key: value for key, value in control.items()
                          if key not in {"arm", "process"}})
        with self.assertRaisesRegex(AssertionError, "belongs to another arm or process"):
            self.check(proof, self.artifacts)

    def test_each_artifact_cannot_be_swapped_with_its_neighbor(self):
        for index in range(len(self.proof)):
            proof = copy.deepcopy(self.proof)
            donor = proof[(index + 1) % len(proof)]
            proof[index].update({key: value for key, value in donor.items()
                                 if key not in {"arm", "process"}})
            with self.subTest(index=index), self.assertRaisesRegex(AssertionError, "belongs to another arm or process"):
                self.check(proof, self.artifacts)

    def test_boolean_process_identity_is_refused(self):
        proof = copy.deepcopy(self.proof)
        next(entry for entry in proof if entry["process"] == 1)["process"] = True
        with self.assertRaisesRegex(AssertionError, "invalid proof process identity"):
            self.check(proof, self.artifacts)

    def test_resealed_invalid_device_values(self):
        name = self.proof[0]["event_artifact"]
        for value in ["/tmp/private", "unexpected", "", None, False]:
            artifacts = copy.deepcopy(self.artifacts)
            event = next(e for e in artifacts[name]["events"] if e["event"] in verifier.LOADED_EVENTS)
            event["device"] = value
            with self.subTest(value=value), self.assertRaisesRegex(AssertionError, "unknown redacted device"):
                self.check(self.proof, artifacts)

    def test_resealed_device_on_unrelated_events(self):
        name = self.proof[0]["event_artifact"]
        for kind in ["other", "calibration", "candidate_rejected"]:
            artifacts = copy.deepcopy(self.artifacts)
            event = artifacts[name]["events"][0]
            event.update(event=kind, device="cpu")
            with self.subTest(event=kind), self.assertRaisesRegex(AssertionError, "device belongs only to loaded events"):
                self.check(self.proof, artifacts)

    def test_each_loaded_device_summary_is_bound(self):
        for index, entry in enumerate(self.proof):
            proof = copy.deepcopy(self.proof)
            lines = proof[index]["loaded_device_evidence"]
            lines[0] = lines[0].replace('device="cpu"', 'device="cuda"').replace('device="coreml"', 'device="cuda"')
            with self.subTest(arm=entry["arm"], process=entry["process"]), self.assertRaisesRegex(AssertionError, "loaded-device summary differs"):
                self.check(proof, self.artifacts)

    def test_resealed_valid_device_change_disagrees_with_summary(self):
        name = self.proof[0]["event_artifact"]
        artifacts = copy.deepcopy(self.artifacts)
        event = next(e for e in artifacts[name]["events"] if e["event"] in verifier.LOADED_EVENTS)
        event["device"] = "cuda"
        with self.assertRaisesRegex(AssertionError, "loaded-device summary differs"):
            self.check(self.proof, artifacts)

    def test_coordinated_coreml_relabeling_is_rejected_after_resealing(self):
        proof = copy.deepcopy(self.proof)
        artifacts = copy.deepcopy(self.artifacts)
        changed = 0
        for entry in proof:
            for event in artifacts[entry["event_artifact"]]["events"]:
                if event.get("device") == "coreml":
                    event["device"] = "cuda"
                    changed += 1
            entry["loaded_device_evidence"] = [line.replace('device="coreml"', 'device="cuda"')
                                                for line in entry["loaded_device_evidence"]]
        self.assertEqual(changed, 12, "counterexample must change every retained CoreML loaded event")
        with self.assertRaisesRegex(AssertionError, "frozen arm/process premise"):
            self.check(proof, artifacts)

    def test_every_frozen_loaded_role_is_independently_pinned(self):
        for index, entry in enumerate(self.proof):
            name = entry["event_artifact"]
            loaded = [e for e in self.artifacts[name]["events"] if e["event"] in verifier.LOADED_EVENTS]
            for position, original in enumerate(loaded):
                proof = copy.deepcopy(self.proof)
                artifacts = copy.deepcopy(self.artifacts)
                event = [e for e in artifacts[name]["events"] if e["event"] in verifier.LOADED_EVENTS][position]
                event["device"] = "cuda"
                proof[index]["loaded_device_evidence"][position] = proof[index]["loaded_device_evidence"][position].replace(
                    f'device="{original["device"]}"', 'device="cuda"')
                with self.subTest(arm=entry["arm"], process=entry["process"], role=original["event"]), self.assertRaisesRegex(AssertionError, "frozen arm/process premise"):
                    self.check(proof, artifacts)

    def test_resealed_valid_model_cannot_be_attributed_to_another_arm(self):
        single = "gpahal/bge-m3-onnx-int8"
        dual = "bge-m3-int8@2b34e84df040034d4b9eabb62383a87c18955822+pplx-0.6b@2c4d510dd4a732063c31a0f70193e35067b51fd8:pool-int8-single-v2-level4"
        for index, entry in enumerate(self.proof):
            if entry["arm"].startswith("main-"):
                continue
            proof = copy.deepcopy(self.proof)
            lines = proof[index]["loaded_device_evidence"]
            position = next(i for i, line in enumerate(lines) if "embedder loaded" in line)
            source, wrong = (dual, single) if entry["arm"] == "dual-cpu" else (single, dual)
            lines[position] = lines[position].replace(f'model="{source}"', f'model="{wrong}"')
            with self.subTest(arm=entry["arm"], process=entry["process"]), self.assertRaisesRegex(AssertionError, "frozen arm metadata"):
                self.check(proof, self.artifacts)

    def test_resealed_reranker_metadata_cannot_change(self):
        for index, entry in enumerate(self.proof):
            for source, wrong in [('tier="accurate"', 'tier="fast"'), ('maximum_tokens=256', 'maximum_tokens=128')]:
                proof = copy.deepcopy(self.proof)
                lines = proof[index]["loaded_device_evidence"]
                position = next(i for i, line in enumerate(lines) if "reranker loaded" in line)
                lines[position] = lines[position].replace(source, wrong)
                with self.subTest(arm=entry["arm"], process=entry["process"], source=source), self.assertRaisesRegex(AssertionError, "frozen arm metadata"):
                    self.check(proof, self.artifacts)

    def test_coordinated_loaded_order_change_is_rejected(self):
        proof = copy.deepcopy(self.proof)
        artifacts = copy.deepcopy(self.artifacts)
        index = next(i for i, p in enumerate(proof) if p["arm"] == "new-auto-persist-hit" and p["process"] == 2)
        events = artifacts[proof[index]["event_artifact"]]["events"]
        positions = [i for i, event in enumerate(events) if event["event"] in verifier.LOADED_EVENTS]
        self.assertEqual(len(positions), 2)
        first, second = positions
        events[first]["event"], events[second]["event"] = events[second]["event"], events[first]["event"]
        events[first]["device"], events[second]["device"] = events[second]["device"], events[first]["device"]
        proof[index]["loaded_device_evidence"].reverse()
        with self.assertRaisesRegex(AssertionError, "frozen arm/process premise"):
            self.check(proof, artifacts)


if __name__ == "__main__":
    unittest.main()
