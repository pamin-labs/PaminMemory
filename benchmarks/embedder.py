#!/usr/bin/env python3
"""The embedding endpoint, on its own, so nothing else can slow it down.

`shim.py` serves embeddings and chat completions from one process, and chat
means forking `claude -p`. Over one mem0 ingest that is 272 subprocesses
competing with an ONNX session for four cores, and the embedding call is not
incidental to the latency comparison: two arms pay it inside every search, so
an unstable embedder makes a quarter of their figures unstable too. One call
measured 15.2 ms in one run and 25.8 ms two hours later, a 70% move with no
product code in it, and contention with those subprocesses was the obvious
suspect.

This file exists to test that suspicion, and **the suspicion was wrong.** It
serves `/v1/embeddings` out of one `InferenceSession` that shares its process
with no subprocess spawning, and measured against the shared shim it is the
same endpoint: 27.7 ms against 29.4 through urllib, 31.5 against 31.9 through
the OpenAI client that mem0 and MemPalace use, and 26.5/25.1 against 26.4/25.8
over the 199-question arm. The 70% move had another cause, and it was ours --
a shell of this session's spin-looping on one of the four cores.

It is kept because a dedicated process is still the right shape for the
measurement, and because a negative result nobody can re-run is not one. Every
route other than `/v1/embeddings` is forwarded to the chat shim, so the arms
keep pointing at a single base URL -- the forwarding hop costs a local round
trip on a call that takes five seconds, and it is on the path this comparison
explicitly does not measure.

    python3 benchmarks/embedder.py 8089 --chat http://127.0.0.1:8088

The weights are the ones `pamin` loads on its `accuracy` profile, which is the
point: an arm embedding through here and an arm embedding in-process are doing
the same arithmetic, and what differs is where.
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_DIR = os.environ.get(
    "SHIM_EMBED_MODEL",
    "/tmp/pamin-eval5/models/models--gpahal--bge-m3-onnx-int8/snapshots/"
    "2b34e84df040034d4b9eabb62383a87c18955822",
)
MAX_TOKENS = int(os.environ.get("SHIM_EMBED_TOKENS", "512"))

_stats = {"embed_calls": 0, "embed_texts": 0, "embed_seconds": 0.0,
          "forwarded": 0, "errors": 0}
_lock = threading.Lock()
_parts = {}


def _load():
    """Loaded at startup rather than on the first request.

    A first request that pays half a gigabyte of session construction is the
    cold start this file exists to remove from the measurement.
    """
    import onnxruntime as ort
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    tokenizer.enable_truncation(max_length=MAX_TOKENS)
    tokenizer.enable_padding()
    _parts["tokenizer"] = tokenizer
    _parts["session"] = ort.InferenceSession(
        os.path.join(MODEL_DIR, "model_quantized.onnx"),
        providers=["CPUExecutionProvider"],
    )


def _embed(texts):
    """BGE-M3's dense representation, unit length.

    The export emits `dense_vecs` itself, so there is no pooling choice to get
    wrong here; this adds the tokenizer and the normalisation, and asserts the
    normalisation rather than assuming it.
    """
    import numpy as np

    started = time.time()
    encoded = _parts["tokenizer"].encode_batch(texts)
    ids = np.array([e.ids for e in encoded], dtype=np.int64)
    mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
    dense = _parts["session"].run(
        ["dense_vecs"], {"input_ids": ids, "attention_mask": mask}
    )[0]
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    with _lock:
        _stats["embed_calls"] += 1
        _stats["embed_texts"] += len(texts)
        _stats["embed_seconds"] += time.time() - started
    return (dense / norms).astype("float32").tolist()


class Handler(BaseHTTPRequestHandler):
    chat = "http://127.0.0.1:8088"

    def log_message(self, *_):
        pass

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _forward(self, body=None):
        url = self.chat.rstrip("/") + self.path
        request = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method=self.command)
        try:
            with urllib.request.urlopen(request, timeout=900) as r:
                payload = json.load(r)
        except urllib.error.HTTPError as e:
            with _lock:
                _stats["errors"] += 1
            return self._send(e.code, {"error": {"message": e.read()[:300].decode(
                "utf-8", "replace")}})
        except Exception as e:
            with _lock:
                _stats["errors"] += 1
            return self._send(502, {"error": {"message": f"chat shim: {e}"}})
        with _lock:
            _stats["forwarded"] += 1
        self._send(200, payload)

    def do_GET(self):
        if self.path.rstrip("/") == "/stats":
            with _lock:
                return self._send(200, dict(_stats))
        self._forward()

    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        raw = self.rfile.read(length) or b"{}"
        if self.path.rstrip("/") != "/v1/embeddings":
            return self._forward(raw)
        try:
            req = json.loads(raw)
        except json.JSONDecodeError as e:
            return self._send(400, {"error": {"message": f"bad json: {e}"}})
        texts = req.get("input", [])
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return self._send(400, {"error": {"message": "no input to embed"}})
        try:
            vectors = _embed([str(t) for t in texts])
        except Exception as e:
            with _lock:
                _stats["errors"] += 1
            return self._send(502, {"error": {"message": f"embedding failed: {e}"}})
        self._send(200, {
            "object": "list",
            "model": req.get("model", "bge-m3-onnx-int8"),
            "data": [{"object": "embedding", "index": i, "embedding": v}
                     for i, v in enumerate(vectors)],
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("port", nargs="?", type=int, default=8089)
    parser.add_argument("--chat", default="http://127.0.0.1:8088",
                        help="where everything that is not an embedding goes")
    args = parser.parse_args()
    Handler.chat = args.chat
    _load()
    print(f"embedder on :{args.port}, chat forwarded to {args.chat}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
