"""CPU HTTP fixture for exercising the production client and process lifecycle."""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from social_jev.transformers import TransformersEngine

engine = TransformersEngine(sys.argv[1])
lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with lock:
            scores = engine.score(payload["input"])
        data = [{"index": index, "data": [score]} for index, score in enumerate(scores)]
        body = json.dumps({"data": list(reversed(data))}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


ThreadingHTTPServer(("127.0.0.1", int(sys.argv[2])), Handler).serve_forever()
