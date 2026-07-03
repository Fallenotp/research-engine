import json
from urllib import request


def test_proxy_json_enforce():
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
  "http://127.0.0.1:8084/v1/chat/completions",
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
