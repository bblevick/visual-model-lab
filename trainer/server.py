"""Loopback-only training controller. No uploads or remote inference service."""

import hashlib, json, os, secrets, signal, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKEN = secrets.token_urlsafe(24)
LOCK = threading.Lock()
JOB = None
ALLOWED = {"http://127.0.0.1:4197", "http://localhost:4197"}


def split_custom(text, target):
    if not isinstance(text, str) or len(text.encode()) > 1_000_000:
        raise ValueError("Dataset must be JSONL under 1 MB")
    rows = []
    seen = set()
    for line in text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        messages = row.get("messages")
        if (
            not isinstance(messages, list)
            or len(messages) != 2
            or [m.get("role") for m in messages] != ["user", "assistant"]
            or any(
                not isinstance(m.get("content"), str)
                or not 1 <= len(m["content"]) <= 600
                for m in messages
            )
        ):
            raise ValueError(
                "Each row needs one user and one assistant message, each 1–600 characters"
            )
        key = messages[0]["content"].strip()
        if key in seen:
            raise ValueError("User prompts must be unique across the dataset")
        seen.add(key)
        rows.append({"messages": messages})
    if not 12 <= len(rows) <= 500:
        raise ValueError("Provide 12–500 unique examples")
    rows.sort(
        key=lambda r: hashlib.sha256(r["messages"][0]["content"].encode()).hexdigest()
    )
    n = len(rows)
    a = int(n * 0.7)
    b = int(n * 0.85)
    target.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", rows[:a]), ("valid", rows[a:b]), ("test", rows[b:])]:
        (target / (name + ".jsonl")).write_text(
            "".join(json.dumps(r) + "\n" for r in part)
        )
    return target


def state():
    global JOB
    if not JOB:
        return {"status": "idle", "log": "", "report": None}
    process = JOB["process"]
    code = process.poll()
    status = (
        "running"
        if code is None
        else (
            "complete"
            if code == 0
            else JOB.get("cancelled") and "cancelled" or "failed"
        )
    )
    log = JOB["out"] / "controller.log"
    report = JOB["out"] / "report.json"
    return {
        "status": status,
        "log": log.read_text(errors="replace")[-16000:] if log.exists() else "",
        "report": (
            json.loads(report.read_text())
            if status == "complete" and report.exists()
            else None
        ),
    }


class Handler(BaseHTTPRequestHandler):
    def send(self, status, value):
        data = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ["/health", "/trainer/health"]:
            return self.send(
                200,
                {
                    "localTrainer": True,
                    "token": TOKEN,
                    "modelAvailable": (
                        ROOT / ".local/base-model/model.safetensors"
                    ).exists(),
                },
            )
        if self.path in ["/status", "/trainer/status"]:
            return self.send(200, state())
        self.send(404, {"error": "Unknown trainer route"})

    def do_POST(self):
        global JOB
        try:
            if self.headers.get("Origin") not in ALLOWED:
                raise PermissionError(
                    "Only the local training interface may start or stop jobs"
                )
            if self.headers.get("Content-Type") != "application/json":
                raise ValueError("Use application/json")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 1_100_000:
                raise ValueError("Request is empty or too large")
            data = json.loads(self.rfile.read(length))
            if not secrets.compare_digest(str(data.get("token", "")), TOKEN):
                raise PermissionError("Trainer token did not match")
            with LOCK:
                if self.path in ["/stop", "/trainer/stop"]:
                    if JOB and JOB["process"].poll() is None:
                        JOB["cancelled"] = True
                        os.killpg(JOB["process"].pid, signal.SIGTERM)
                    return self.send(200, {"stopped": True})
                if self.path not in ["/train", "/trainer/train"]:
                    return self.send(404, {"error": "Unknown trainer route"})
                if JOB and JOB["process"].poll() is None:
                    raise ValueError("A training job is already running")
                steps = data.get("steps", 60)
                rate = data.get("learningRate", 0.0003)
                if (
                    type(steps) is not int
                    or steps not in [20, 60, 120]
                    or rate not in [0.0001, 0.0003, 0.001]
                ):
                    raise ValueError("Unsupported training parameters")
                out = ROOT / ".local/runs" / str(time.time_ns())
                out.mkdir(parents=True)
                dataset = ROOT / "trainer/data"
                if data.get("dataset") is not None:
                    dataset = split_custom(data["dataset"], out / "data")
                command = [
                    sys.executable,
                    str(ROOT / "trainer/run.py"),
                    "--steps",
                    str(steps),
                    "--learning-rate",
                    str(rate),
                    "--data",
                    str(dataset),
                    "--out",
                    str(out),
                ]
                log = (out / "controller.log").open("w")
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                log.close()
                JOB = {"process": process, "out": out}
                self.send(202, {"started": True})
        except PermissionError as e:
            self.send(403, {"error": str(e)})
        except Exception as e:
            self.send(400, {"error": str(e)})

    def log_message(self, *args):
        pass


if __name__ == "__main__":

    def terminate(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    print("MLX training controller: http://127.0.0.1:8765 (loopback only)", flush=True)
    try:
        ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
    finally:
        if JOB and JOB["process"].poll() is None:
            os.killpg(JOB["process"].pid, signal.SIGTERM)
