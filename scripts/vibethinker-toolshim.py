#!/usr/bin/env python3
"""vibethinker-toolshim.py — OpenAI-compatible recovery proxy for VibeThinker tool calls.

VibeThinker reliably produces correct tool *intent* — the right tool name and valid
arguments as a JSON object — but wraps it in a hallucinated tag (<fulfil>, <function call>,
…) instead of the <tool_call> llama.cpp greps for, so llama.cpp never populates `tool_calls`
and clients (pi) just see text. This shim sits between the client and llama-server:

    client (pi)  ->  this shim (:8088)  ->  llama-server (:8080)

It forwards every request, and when a chat response has NO native `tool_calls` but its
content carries a recoverable `{"name","arguments"}` object whose name is one of the
*declared* tools, it rewrites the response into a proper `tool_calls` so the client executes
it. Names that don't match a declared tool are rejected (the model sometimes invents names).

REQUIRES a reasoning-budget-capped server (e.g. REASONING_BUDGET=600 run-vibethinker.sh):
without it the model ruminates past the token limit and emits no call to recover. Misses are
logged to stderr so you can see what format slipped through.

Usage:
    python scripts/vibethinker-toolshim.py            # listen :8088, upstream :8080
    UPSTREAM=http://127.0.0.1:8080 PORT=8088 python scripts/vibethinker-toolshim.py
Then point pi's provider baseUrl at http://127.0.0.1:8088/v1.
"""
import json, os, sys, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get("UPSTREAM", "http://127.0.0.1:8080").rstrip("/")
PORT     = int(os.environ.get("PORT", "8088"))
HOST     = os.environ.get("HOST", "127.0.0.1")
_n = [0]


def log(*a):
    print("[toolshim]", *a, file=sys.stderr, flush=True)


def recover_tool_call(content, valid_names):
    """First balanced {...} that JSON-decodes to {name in valid_names, arguments}."""
    if not content:
        return None
    dec = json.JSONDecoder()
    i = 0
    while i < len(content):
        j = content.find("{", i)
        if j < 0:
            break
        try:
            obj, _ = dec.raw_decode(content, j)
        except json.JSONDecodeError:
            i = j + 1
            continue
        if isinstance(obj, dict) and "arguments" in obj:
            # the model uses "name" under the OpenAI tools API and "function" under
            # pi's prompt framing — accept either as the tool-name key.
            name = obj.get("name") or obj.get("function")
            if name in valid_names:
                args = obj["arguments"]
                if not isinstance(args, str):
                    args = json.dumps(args)
                return name, args
        i = j + 1
    return None


def maybe_recover(resp, valid_names):
    """Mutate resp in place if a tool call is recovered. Returns True if it did."""
    try:
        ch = resp["choices"][0]
        msg = ch["message"]
    except (KeyError, IndexError):
        return False
    if not valid_names or msg.get("tool_calls"):
        return False
    rec = recover_tool_call(msg.get("content") or "", valid_names)
    if not rec:
        c = (msg.get("content") or "")
        if any(k in c for k in ('"name"', "<function", "<tool", "arguments")):
            log("MISS: looked like a call but unrecoverable:", repr(c[:160]))
        return False
    name, args = rec
    _n[0] += 1
    msg["tool_calls"] = [{"id": f"call_vtshim_{_n[0]}", "type": "function",
                          "function": {"name": name, "arguments": args}}]
    msg["content"] = None
    ch["finish_reason"] = "tool_calls"
    log(f"RECOVERED tool_call -> {name}({args[:60]}...)")
    return True


def sse(obj):
    return f"data: {json.dumps(obj)}\n\n".encode()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        self._passthrough("GET", None)

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
        if self.path.rstrip("/").endswith("/chat/completions"):
            self._chat(body)
        else:
            self._passthrough("POST", body)

    def _passthrough(self, method, body):
        try:
            req = urllib.request.Request(UPSTREAM + self.path, data=body, method=method)
            for k, v in self.headers.items():
                if k.lower() not in ("host", "content-length", "connection"):
                    req.add_header(k, v)
            r = urllib.request.urlopen(req, timeout=900)
            data = r.read()
            self.send_response(r.status)
            self.send_header("Content-Type", r.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self.send_error(502, str(e))

    def _chat(self, body):
        try:
            req_obj = json.loads(body)
        except Exception as e:
            self.send_error(400, f"bad json: {e}")
            return
        want_stream = bool(req_obj.get("stream"))
        valid = {t["function"]["name"] for t in req_obj.get("tools", [])
                 if isinstance(t, dict) and t.get("type") == "function"}
        upstream_req = dict(req_obj)
        upstream_req["stream"] = False   # buffer upstream so we can recover on full content
        try:
            r = urllib.request.urlopen(urllib.request.Request(
                UPSTREAM + "/v1/chat/completions",
                data=json.dumps(upstream_req).encode(),
                headers={"Content-Type": "application/json"}), timeout=900)
            resp = json.loads(r.read())
        except Exception as e:
            self.send_error(502, str(e))
            return
        maybe_recover(resp, valid)
        if want_stream:
            self._stream_down(resp)
        else:
            out = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    def _stream_down(self, resp):
        ch = resp["choices"][0]
        msg = ch["message"]
        rid = resp.get("id", "vtshim")
        model = resp.get("model", "vibethinker")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        base = {"id": rid, "object": "chat.completion.chunk", "model": model}

        def chunk(delta, finish=None):
            return sse({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})

        self.wfile.write(chunk({"role": "assistant"}))
        if msg.get("reasoning_content"):
            self.wfile.write(chunk({"reasoning_content": msg["reasoning_content"]}))
        if msg.get("tool_calls"):
            for k, tc in enumerate(msg["tool_calls"]):
                self.wfile.write(chunk({"tool_calls": [{
                    "index": k, "id": tc["id"], "type": "function",
                    "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}]}))
        if msg.get("content"):
            self.wfile.write(chunk({"content": msg["content"]}))
        self.wfile.write(chunk({}, finish=ch.get("finish_reason") or "stop"))
        self.wfile.write(b"data: [DONE]\n\n")


if __name__ == "__main__":
    log(f"listening on http://{HOST}:{PORT}  ->  upstream {UPSTREAM}")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
