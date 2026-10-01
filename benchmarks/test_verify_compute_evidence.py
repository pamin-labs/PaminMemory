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
        for arm, block in ((arm, block) for arm in ("main-cpu", "new-cpu") for block in range(3)):
            raw = dict(self.raw)
            filename = f"{arm}-{block}.jsonl"
            raw[filename] = copy.deepcopy(raw[filename])
            query = next(r for r in raw[filename] if r["kind"] == "query")
            query["ranked"] = ["changed-ranking"]
            with self.subTest(block=block), self.assertRaisesRegex(AssertionError, "default CPU ranking differs"):
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

    def test_resealed_top_level_extra_text_is_rejected(self):
        name = self.proof[0]["event_artifact"]
        artifacts = copy.deepcopy(self.artifacts)
        artifacts[name]["internal_note"] = "unapproved free text"
        with self.assertRaisesRegex(AssertionError, "top-level fields"):
            self.check(self.proof, artifacts)

    def test_resealed_top_level_types_and_marker_are_rejected(self):
        name = self.proof[0]["event_artifact"]
        for field, value in [("redaction", "changed marker"), ("redaction", 1),
                             ("source_sha256", 1), ("events", {})]:
            artifacts = copy.deepcopy(self.artifacts)
            artifacts[name][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(AssertionError):
                self.check(self.proof, artifacts)

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
            proof[index]["loaded_device_events"][0]["device"] = "cuda"
            with self.subTest(arm=entry["arm"], process=entry["process"]), self.assertRaisesRegex(AssertionError, "loaded-device summary differs"):
                self.check(proof, self.artifacts)

    def test_resealed_valid_device_change_disagrees_with_summary(self):
        name = self.proof[0]["event_artifact"]
        artifacts = copy.deepcopy(self.artifacts)
        event = next(e for e in artifacts[name]["events"] if e["event"] in verifier.LOADED_EVENTS)
        event["device"] = "cuda"
        with self.assertRaisesRegex(AssertionError, "loaded-device summary differs"):
            self.check(self.proof, artifacts)

    def test_resealed_valid_device_and_summary_still_need_the_measured_arm(self):
        proof = copy.deepcopy(self.proof)
        artifacts = copy.deepcopy(self.artifacts)
        for entry in proof:
            for event in artifacts[entry["event_artifact"]]["events"]:
                if event.get("device") == "coreml":
                    event["device"] = "cuda"
            entry["loaded_device_events"] = [{k: v for k, v in event.items() if k != "line"}
                for event in artifacts[entry["event_artifact"]]["events"] if event["event"] in verifier.LOADED_EVENTS]
        with self.assertRaisesRegex(AssertionError, "do not match the measured arm"):
            self.check(proof, artifacts)

    def test_resealed_changed_model_or_tier_is_not_proof(self):
        for kind, field, value, message in [("embedder_loaded", "model", "wrong-model", "wrong measured model"),
                ("reranker_loaded", "tier", "fast", "wrong measured reranker settings"),
                ("reranker_loaded", "maximum_tokens", 512, "wrong measured reranker settings")]:
            proof = copy.deepcopy(self.proof)
            artifacts = copy.deepcopy(self.artifacts)
            entry = next(p for p in proof if p["arm"] == "dual-cpu")
            event = next(e for e in artifacts[entry["event_artifact"]]["events"] if e["event"] == kind)
            event[field] = value
            entry["loaded_device_events"] = [{k: v for k, v in e.items() if k != "line"}
                for e in artifacts[entry["event_artifact"]]["events"] if e["event"] in verifier.LOADED_EVENTS]
            with self.subTest(field=field), self.assertRaisesRegex(AssertionError, message):
                self.check(proof, artifacts)


if __name__ == "__main__":
    unittest.main()
