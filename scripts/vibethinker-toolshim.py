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
import json, os, re, sys, textwrap, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get("UPSTREAM", "http://127.0.0.1:8080").rstrip("/")
PORT     = int(os.environ.get("PORT", "8088"))
HOST     = os.environ.get("HOST", "127.0.0.1")
_n = [0]


def log(*a):
    print("[toolshim]", *a, file=sys.stderr, flush=True)


def tool_specs(tools_field):
    """[{name, props:set, required:set}] from the request's `tools` array."""
    specs = []
    for t in tools_field or []:
        if not isinstance(t, dict) or t.get("type") != "function":
            continue
        fn = t.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        params = fn.get("parameters") or {}
        specs.append({"name": name,
                      "props": set((params.get("properties") or {}).keys()),
                      "required": set(params.get("required") or [])})
    return specs


def infer_tool(args, specs):
    """Pick the unique declared tool whose schema best fits a BARE args object.
    Hard requirement: the tool's required params are all present. Among those, the
    fewest 'foreign' keys (keys not in the tool's properties) wins; ties → give up."""
    if not isinstance(args, dict) or not args:
        return None
    keys = set(args.keys())
    cands = []
    for s in specs:
        if not s["required"] <= keys:
            continue
        if not s["required"] and not (keys and keys <= s["props"]):
            continue  # a no-required tool would match anything — only take an exact fit
        cands.append((len(keys - s["props"]), s["name"]))
    if not cands:
        return None
    cands.sort()
    if len(cands) == 1 or cands[0][0] < cands[1][0]:
        return cands[0][1]
    return None


# Terminate the function block tolerantly: VibeThinker frequently MISMATCHES its
# XML — closing </content></args></call> but omitting </function>, or truncating
# mid-content. So accept </function>, </call>, or end-of-string as the boundary.
_FUNC_RE   = re.compile(r"<function\b([^>]*)>(.*?)(?:</function>|</call>|$)", re.DOTALL)
_ATTR_NAME = re.compile(r'name\s*=\s*"([^"]+)"')
_EL_NAME   = re.compile(r"<name>\s*([A-Za-z_]\w*)\s*</name>")


def _coerce(raw):
    """Turn an XML element's inner text into a Python value. Multi-line bodies
    (e.g. a write's `content`) are dedented and kept as a string; scalars like
    null / 0 / true / "x" are JSON-decoded so types are right."""
    if "\n" in raw.strip("\n"):
        return textwrap.dedent(raw.strip("\n")).rstrip()
    t = raw.strip()
    try:
        return json.loads(t)
    except Exception:
        return t


def recover_xml(content, specs):
    """Recover a tool call from VibeThinker's two natural XML dialects:
      (A) <function name="bash"><args><command>ls</command></args></function>
      (B) <function><name>bash</name><arguments><object><command>ls</command>...
    Schema-driven: once the tool name is known, pull each DECLARED param out of
    the block by its own <tag>…</tag>. Tolerant of unescaped code in <content>."""
    by_name = {s["name"]: s for s in specs}
    for mo in _FUNC_RE.finditer(content):
        attrs, inner = mo.group(1), mo.group(2)
        m = _ATTR_NAME.search(attrs) or _EL_NAME.search(inner)
        if not m:
            continue
        spec = by_name.get(m.group(1))
        if not spec:
            continue
        args = {}
        for prop in spec["props"]:
            # tolerant close: a truncated/mismatched body still yields the value
            pm = re.search(r"<%s>(.*?)(?:</%s>|$)" % (re.escape(prop), re.escape(prop)),
                           inner, re.DOTALL)
            if pm and pm.group(1).strip():
                args[prop] = _coerce(pm.group(1))
        if spec["required"] <= set(args.keys()):
            return spec["name"], json.dumps(args)
    return None


def recover_tool_call(content, specs):
    """Recover (name, args_json_str) from the model's free-text tool call. Handles:
      (a) inline {name|function, arguments}     — OpenAI / pi 'function' framing
      (b) <function>NAME</function> + args      — pi's XML framing with a name
      (c) a BARE args object, name absent       — infer the tool from the arg shape
      (d) <function name=…>/<function><name>…   — schema-driven XML (recover_xml)
    """
    if not content or not specs:
        return None
    names = {s["name"] for s in specs}
    m = re.search(r"<function>\s*([A-Za-z_]\w*)\s*</function>", content)
    tag_name = m.group(1) if m and m.group(1) in names else None
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
        if isinstance(obj, dict):
            name = obj.get("name") or obj.get("function")
            if name in names and "arguments" in obj:                      # (a)
                a = obj["arguments"]
                return name, a if isinstance(a, str) else json.dumps(a)
            if "arguments" not in obj and "name" not in obj and "function" not in obj:
                inferred = tag_name or infer_tool(obj, specs)             # (b)/(c)
                if inferred:
                    return inferred, json.dumps(obj)
        i = j + 1
    return recover_xml(content, specs)                                     # (d)


def maybe_recover(resp, specs):
    """Mutate resp in place if a tool call is recovered. Returns True if it did."""
    try:
        ch = resp["choices"][0]
        msg = ch["message"]
    except (KeyError, IndexError):
        return False
    if not specs or msg.get("tool_calls"):
        return False
    rec = recover_tool_call(msg.get("content") or "", specs)
    if not rec:
        c = (msg.get("content") or "")
        if any(k in c for k in ('"name"', "<function", "<tool", "arguments", "args-json")):
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


# A measured system-prompt nudge. OPT-IN (VT_SYSTEM_NUDGE=1), default OFF.
# In a raw *clean-context* prompt sweep this framing took the 3B's bash-call rate
# to 4/4 (vs 3/4 baseline) and a careless "emit the call first" variant dropped it
# to 2/4 — framing matters. BUT in the real DEEP pi context (full system prompt +
# accumulated turns) an A/B showed no clear win: bash-run 1/4→2/4 (noise) and write
# actually dropped 4/4→2/4. The deep-context bash-simulation is *contextual*, not
# promptable — keep sessions short instead. Left as a knob for experimentation;
# the shim owns the upstream request, so it can inject this whatever pi sends.
NUDGE = ("Important: your text replies are shown to the user but are NEVER executed. "
         "The ONLY way to actually run a command, read a file, or change the filesystem is to "
         "emit a tool call. Any command output you write yourself without calling a tool is a "
         "hallucination — to run something, call the bash tool with the exact command.")


def inject_nudge(req):
    if os.environ.get("VT_SYSTEM_NUDGE", "0") != "1" or not req.get("tools"):
        return
    msgs = req.get("messages")
    if not isinstance(msgs, list):
        return
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "system":
            c = m.get("content")
            if isinstance(c, str) and NUDGE not in c:
                m["content"] = c.rstrip() + "\n\n" + NUDGE
            elif isinstance(c, list) and not any(
                    isinstance(p, dict) and p.get("text") == NUDGE for p in c):
                c.append({"type": "text", "text": NUDGE})
            return
    msgs.insert(0, {"role": "system", "content": NUDGE})


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
        specs = tool_specs(req_obj.get("tools"))
        upstream_req = dict(req_obj)
        upstream_req["stream"] = False   # buffer upstream so we can recover on full content
        inject_nudge(upstream_req)       # measured anti-simulation system nudge (VT_SYSTEM_NUDGE=0 off)
        try:
            r = urllib.request.urlopen(urllib.request.Request(
                UPSTREAM + "/v1/chat/completions",
                data=json.dumps(upstream_req).encode(),
                headers={"Content-Type": "application/json"}), timeout=900)
            resp = json.loads(r.read())
        except Exception as e:
            self.send_error(502, str(e))
            return
        maybe_recover(resp, specs)
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
