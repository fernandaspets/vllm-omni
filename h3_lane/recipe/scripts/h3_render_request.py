#!/usr/bin/env python3
"""One canary request against a served H3 lane, with the artifact's size and digest as the output.

The point is the receipt, not the render: the request is only interesting if the clip it produces can
be compared byte-for-byte against a recorded one.

Env: SOLREFS, STEPS, SECONDS, SEED, FLOW_SHIFT, AUDIO_SHIFT, OUT_DIR, REFS (comma separated),
     PROMPT_FILE, AUDIO_FILE, PORT
Prints one JSON object: wall_s, ok, path, bytes, sha256, and the engine time if the response has it.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import urllib.request
import uuid
from pathlib import Path

S = Path(os.environ.get("SOLREFS", "/home/giga/comfyui/models/solrefs"))
STEPS = os.environ.get("STEPS", "4")
SECONDS = float(os.environ.get("SECONDS", "5"))
SEED = os.environ.get("SEED", "20261005")
FLOW_SHIFT = os.environ.get("FLOW_SHIFT", "12.0")
AUDIO_SHIFT = os.environ.get("AUDIO_SHIFT", "3.0")
PORT = os.environ.get("PORT", "8000")
OUT_DIR = Path(os.environ.get("OUT_DIR", "/tmp"))
REFS = [p for p in os.environ.get(
    "REFS", f"{S}/rocco_best.jpg,{S}/roxy_1111.jpg").split(",") if p]
PROMPT_FILE = Path(os.environ.get("PROMPT_FILE", S / "prompt.txt"))
AUDIO_FILE = Path(os.environ.get("AUDIO_FILE", S / "duo_dialogue_32k.wav"))


def field(name: str, value: str) -> bytes:
    return (f'--{B}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n').encode()


def file_part(name: str, filename: str, content_type: str, data: bytes) -> bytes:
    head = (f'--{B}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n")
    return head.encode() + data + b"\r\n"


B = "----h3" + uuid.uuid4().hex
audio_ref = OUT_DIR / "audio_ref.json"
audio_ref.write_text(json.dumps({
    "audio_url": "data:audio/wav;base64," + base64.b64encode(AUDIO_FILE.read_bytes()).decode()
}))

body = b"".join(field(k, v) for k, v in [
    ("aspect_ratio", "16:9"), ("width", "1344"), ("height", "768"), ("fps", "24"),
    ("seconds", str(int(SECONDS))), ("flow_shift", FLOW_SHIFT),
    ("num_inference_steps", STEPS), ("seed", SEED),
])
body += field("prompt", PROMPT_FILE.read_text())
body += field("audio_reference", audio_ref.read_text())
for ref in REFS:
    p = Path(ref)
    body += file_part("input_references", p.name, "image/jpeg", p.read_bytes())
body += field("extra_params", json.dumps(
    {"task": "ref2va", "duration": SECONDS, "audio_flow_shift": float(AUDIO_SHIFT)}))
body += f"--{B}--\r\n".encode()

req = urllib.request.Request(
    f"http://127.0.0.1:{PORT}/v1/videos/sync", data=body,
    headers={"Content-Type": "multipart/form-data; boundary=" + B})

t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=14400) as r:
        out = json.load(r)
    ok = True
except urllib.error.HTTPError as exc:  # the body is the diagnosis; never swallow it
    out, ok = {"error": repr(exc), "status": exc.code,
               "body": exc.read().decode("utf-8", "replace")[:1200]}, False
except Exception as exc:  # noqa: BLE001 - the receipt must say what failed, not raise
    body = ""
    if hasattr(exc, "read"):
        try:
            body = exc.read().decode("utf-8", "replace")[:2000]
        except Exception:
            body = "<unreadable>"
    out, ok = {"error": repr(exc), "body": body}, False
wall = time.time() - t0

result = {"wall_s": round(wall, 3), "ok": ok, "arm_steps": STEPS, "seconds": SECONDS}
if ok:
    path = None
    for key in ("path", "output_path", "video_path", "file"):
        if isinstance(out, dict) and out.get(key):
            path = out[key]
            break
    if path is None and isinstance(out, dict):
        for v in out.values():
            if isinstance(v, str) and v.endswith(".mp4"):
                path = v
    result["path"] = path
    result["engine_s"] = (out.get("engine_time_s") or out.get("inference_time")
                          or (out.get("timings") or {}).get("engine_s")) if isinstance(out, dict) else None
    if path and Path(path).is_file():
        data = Path(path).read_bytes()
        result["bytes"] = len(data)
        result["sha256"] = hashlib.sha256(data).hexdigest()
result["response"] = {k: v for k, v in out.items() if k != "path"} if isinstance(out, dict) else out
print(json.dumps(result))
