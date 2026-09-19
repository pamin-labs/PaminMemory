#!/usr/bin/env python3
"""An OpenAI-compatible endpoint backed by `claude -p`.

Every system in the comparison has to answer with the same model, or the
comparison measures the models. The competitors all speak the OpenAI chat
API, and this container has no API key but does have the Claude Code CLI, so
this bridges the two.

Deliberately small: chat completions only, no streaming, no tools. Anything a
memory system needs beyond that should be noticed rather than silently
emulated.
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = os.environ.get("SHIM_MODEL", "sonnet")
TIMEOUT = int(os.environ.get("SHIM_TIMEOUT", "180"))
LOG = os.environ.get("SHIM_LOG", "/tmp/bench/shim.log")

_stats = {"calls": 0, "errors": 0, "seconds": 0.0,
          "embed_calls": 0, "embed_texts": 0, "embed_seconds": 0.0,
          # Reported by the CLI rather than estimated. Every call also
          # carries this session's own cached context, which is the same
          # for every caller -- so the totals are inflated by a constant
          # and the difference between callers is what can be read.
          "input_tokens": 0, "output_tokens": 0,
          "cache_creation_tokens": 0, "cache_read_tokens": 0,
          "cost_usd": 0.0}
_lock = threading.Lock()


def _flatten(messages):
    """OpenAI messages -> one prompt.

    `claude -p` takes a single prompt, so roles are marked in the text. A
    system message becomes a leading instruction, which is how the CLI's own
    -p mode is meant to be driven.
    """
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):  # OpenAI content-parts form
            content = "".join(
                c.get("text", "") for c in content if isinstance(c, dict)
            )
        if role == "system":
            parts.append(f"[instructions]\n{content}")
        elif role == "assistant":
            parts.append(f"[assistant]\n{content}")
        else:
            parts.append(f"[user]\n{content}")
    return "\n\n".join(parts)


def _call(prompt, want_json):
    if want_json:
        prompt += (
            "\n\n[output]\nReply with JSON only. No prose, no code fence, "
            "nothing before or after the JSON."
        )
    started = time.time()
    try:
        out = subprocess.run(
            ["claude", "-p", "--model", MODEL, "--output-format", "json", prompt],
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        with _lock:
            _stats["errors"] += 1
        raise RuntimeError(f"claude -p timed out after {TIMEOUT}s")
    elapsed = time.time() - started
    with _lock:
        _stats["calls"] += 1
        _stats["seconds"] += elapsed
        if out.returncode != 0:
            _stats["errors"] += 1
    if out.returncode != 0:
        raise RuntimeError(f"claude -p exited {out.returncode}: {out.stderr[:400]}")

    # `--output-format json` wraps the answer and reports what it cost. A
    # comparison of memory systems that cannot say what each one spends is
    # missing the half of the claim that says "less".
    text = out.stdout.strip()
    try:
        envelope = json.loads(text)
        usage = envelope.get("usage") or {}
        with _lock:
            _stats["input_tokens"] += usage.get("input_tokens", 0)
            _stats["output_tokens"] += usage.get("output_tokens", 0)
            _stats["cache_creation_tokens"] += usage.get("cache_creation_input_tokens", 0)
            _stats["cache_read_tokens"] += usage.get("cache_read_input_tokens", 0)
            _stats["cost_usd"] += envelope.get("total_cost_usd", 0.0) or 0.0
        text = (envelope.get("result") or "").strip()
    except (json.JSONDecodeError, AttributeError):
        # An older CLI, or a failure that printed prose. The answer is still
        # whatever came back; the usage for that call is simply not counted,
        # which is better than counting a guess.
        pass

    if want_json:
        # The CLI sometimes fences JSON however firmly it is asked not to.
        fence = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.S)
        if fence:
            text = fence.group(1).strip()
    with open(LOG, "a") as f:
        f.write(json.dumps({"ms": int(elapsed * 1000), "out": len(text)}) + "\n")
    return text


# The same weights PaminMemory embeds with, loaded from the same cache: the
# comparison is between memory systems, and it stops being that the moment one
# of them gets a different embedder. This is the identical ONNX file, not
# another export of the same model.
MODEL_DIR = os.environ.get(
    "SHIM_EMBED_DIR",
    "/tmp/pamin-eval5/models/models--gpahal--bge-m3-onnx-int8/snapshots/"
    "2b34e84df040034d4b9eabb62383a87c18955822",
)
MAX_TOKENS = int(os.environ.get("SHIM_EMBED_TOKENS", "512"))

_embedder = {}
_embed_lock = threading.Lock()


def _load_embedder():
    """Loads once, on the first request that needs it.

    Half a gigabyte of session state, and a run that only uses chat completions
    should not pay for it.
    """
    with _embed_lock:
        if _embedder:
            return _embedder
        import onnxruntime as ort
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
        tokenizer.enable_truncation(max_length=MAX_TOKENS)
        tokenizer.enable_padding()
        _embedder["tokenizer"] = tokenizer
        _embedder["session"] = ort.InferenceSession(
            os.path.join(MODEL_DIR, "model_quantized.onnx"),
            providers=["CPUExecutionProvider"],
        )
        return _embedder


def _embed(texts):
    """BGE-M3's dense representation, unit length.

    The export emits `dense_vecs` itself, so there is no pooling choice to get
    wrong here -- the only thing this adds is the tokenizer and the
    normalisation, and the normalisation is asserted rather than assumed.
    """
    import numpy as np

    parts = _load_embedder()
    started = time.time()
    encoded = parts["tokenizer"].encode_batch(texts)
    ids = np.array([e.ids for e in encoded], dtype=np.int64)
    mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
    dense = parts["session"].run(
        ["dense_vecs"], {"input_ids": ids, "attention_mask": mask}
    )[0]
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    # Counted, because "which arm paid for how much inference" is a result and
    # not an implementation detail: one of the arms calls a model on its write
    # path and the others do not.
    with _lock:
        _stats["embed_calls"] += 1
        _stats["embed_texts"] += len(texts)
        _stats["embed_seconds"] += time.time() - started
    return (dense / norms).astype("float32").tolist()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):  # keep stderr for real problems
        pass

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            self._send(200, {"object": "list", "data": [{"id": MODEL, "object": "model"}]})
        elif self.path.rstrip("/") == "/stats":
            with _lock:
                self._send(200, dict(_stats))
        else:
            self._send(404, {"error": {"message": f"no route {self.path}"}})

    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as e:
            return self._send(400, {"error": {"message": f"bad json: {e}"}})

        route = self.path.rstrip("/")

        if route == "/v1/embeddings":
            texts = req.get("input", [])
            if isinstance(texts, str):
                texts = [texts]
            if not texts:
                return self._send(400, {"error": {"message": "no input to embed"}})
            try:
                vectors = _embed([str(t) for t in texts])
            except Exception as e:
                return self._send(502, {"error": {"message": f"embedding failed: {e}"}})
            return self._send(
                200,
                {
                    "object": "list",
                    "model": req.get("model", "bge-m3-onnx-int8"),
                    "data": [
                        {"object": "embedding", "index": i, "embedding": v}
                        for i, v in enumerate(vectors)
                    ],
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                },
            )

        if route != "/v1/chat/completions":
            return self._send(
                404,
                {"error": {"message": f"this shim serves chat completions and embeddings, got {self.path}"}},
            )

        fmt = req.get("response_format") or {}
        want_json = fmt.get("type") in ("json_object", "json_schema")
        try:
            text = _call(_flatten(req.get("messages", [])), want_json)
        except Exception as e:  # surfaced to the caller rather than swallowed
            return self._send(502, {"error": {"message": str(e)}})

        self._send(
            200,
            {
                "id": f"chatcmpl-shim-{int(time.time()*1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.get("model", MODEL),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                # Real counts are not available through the CLI. Zero rather
                # than a guess, so nothing downstream reports invented usage.
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
        )


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8088
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    print(f"shim on :{port} -> claude -p --model {MODEL}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
