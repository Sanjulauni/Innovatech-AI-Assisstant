"""A stand-in for llama.cpp's llama-server, run as a real process by the tests.

It accepts llama-server's options, answers ``/health`` with 503 while "loading" and
200 afterwards, and answers OpenAI-style chat requests (plain and streamed) with a
fixed reply. Extra options control its behaviour:

    --fake-load-seconds N   time to "load the model" (default 0.2)
    --fake-exit-code N      print an error and exit with this code instead of serving
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPLY = ["The local model ", "says hi ", "[1]."]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--fake-load-seconds", type=float, default=0.2)
    parser.add_argument("--fake-exit-code", type=int)
    args, _unknown = parser.parse_known_args()  # e.g. --ctx-size

    print(f"loading model {args.model}", flush=True)
    if args.fake_exit_code is not None:
        print("error: failed to load model", flush=True)
        sys.exit(args.fake_exit_code)

    ready_at = time.monotonic() + args.fake_load_seconds

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args) -> None:  # keep test output quiet
            pass

        def _json(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                loading = time.monotonic() < ready_at
                self._json(503 if loading else 200, {"status": "loading" if loading else "ok"})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            request = json.loads(self.rfile.read(length) or b"{}")
            if self.path != "/v1/chat/completions":
                self._json(404, {"error": "not found"})
            elif request.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for piece in REPLY:
                    chunk = {
                        "id": "1",
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": "local",
                        "choices": [{"index": 0, "delta": {"content": piece}}],
                    }
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                self._json(200, {
                    "id": "1",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "local",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "".join(REPLY)},
                        "finish_reason": "stop",
                    }],
                })  # fmt: skip

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"server listening on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
