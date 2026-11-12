import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 (stdlib API)
        if self.path.rstrip("/") != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length") or "0")
        body = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(body.decode("utf-8") or "{}")
        except Exception:
            payload = {}

        # Minimal OpenAI-compatible response shape
        reply = {
            "id": "mock-1",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"mock-ok model={payload.get('model','')} msgs={len(payload.get('messages') or [])}",
                    },
                    "finish_reason": "stop",
                }
            ],
        }

        data = json.dumps(reply).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):  # noqa: A002 (stdlib API)
        # Keep output quiet in packaging workflows.
        return


def main() -> None:
    server = HTTPServer(("127.0.0.1", 8999), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()

