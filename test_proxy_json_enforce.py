import json
import socket
from urllib import request

import pytest


PROXY_HOST = "127.0.0.1"
PROXY_PORT = 8084
PROXY_URL = f"http://{PROXY_HOST}:{PROXY_PORT}/v1/chat/completions"


def _mlx_token_proxy_ready() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex((PROXY_HOST, PROXY_PORT)) == 0


def test_proxy_json_enforce():
 if not _mlx_token_proxy_ready():
  pytest.skip(f"MLX token proxy at http://{PROXY_HOST}:{PROXY_PORT} is unreachable")
 payload = {
  "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
  "stream": False,
  "messages": [
   {
    "role": "user",
    "content": "Extract {company, amount_usd, investor, date, ceo} from: Steel raises $5M led by Acme on June 1 2026; CEO is Jane Doe.",
   }
  ],
 }
 req = request.Request(
  PROXY_URL,
  data=json.dumps(payload).encode("utf-8"),
  headers={"content-type": "application/json", "X-Webread-JSON": "1"},
  method="POST",
 )
 with request.urlopen(req, timeout=120) as resp:
  assert resp.status == 200
  resp_data = json.loads(resp.read().decode("utf-8"))
 content = resp_data["choices"][0]["message"]["content"]
 parsed = json.loads(content)
 print(parsed)
 assert {"company", "amount_usd", "investor", "date", "ceo"} <= set(parsed)
