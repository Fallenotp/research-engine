#!/usr/bin/env python3
import asyncio
import json
import time
from datetime import datetime

import aiohttp
import aiohttp.web

LISTEN_PORT = 8084
BACKEND_URL = "http://127.0.0.1:8083"
JSON_ONLY_SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are a JSON extraction engine. Output ONLY one valid JSON object. "
        "No prose, no markdown fences."
    ),
}


class ThinkStripper:
    def __init__(self) -> None:
        self.buffer = ""
        self.in_think = False

    def feed(self, text: str) -> str:
        self.buffer += text
        result: list[str] = []
        i = 0
        while i < len(self.buffer):
            if not self.in_think:
                idx = self.buffer.find("<think>", i)
                if idx == -1:
                    if i < len(self.buffer) - 6:
                        result.append(self.buffer[i : len(self.buffer) - 6])
                        self.buffer = self.buffer[len(self.buffer) - 6 :]
                    break
                result.append(self.buffer[i:idx])
                self.in_think = True
                i = idx + 7
                continue

            idx = self.buffer.find("</think>", i)
            if idx == -1:
                self.buffer = self.buffer[i:]
                break
            self.in_think = False
            i = idx + 8

        return "".join(result)

    def flush(self) -> str:
        return "" if self.in_think else self.buffer


def _json_only_body(req_data):
    req_copy = dict(req_data)
    messages = req_copy.get("messages", [])
    if not isinstance(messages, list):
        messages = []
    req_copy["messages"] = [JSON_ONLY_SYSTEM_MESSAGE, *messages]
    return req_copy


def _clean_json_only_content(content: str) -> str:
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:].lstrip()
    elif content.startswith("```"):
        content = content[3:].lstrip()
    if content.endswith("```"):
        content = content[:-3].rstrip()
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end >= start:
        content = content[start : end + 1]
    return content


async def handle_chat_completions(request):
    start_time = time.monotonic()
    try:
        body = await request.read()
    except Exception as exc:
        return aiohttp.web.json_response(
            {"error": f"Failed to read request: {exc}"},
            status=400,
        )

    try:
        req_data = json.loads(body)
    except json.JSONDecodeError:
        req_data = {}

    force_json = request.headers.get("X-Webread-JSON") == "1"
    if force_json:
        req_data = _json_only_body(req_data)
        body = json.dumps(req_data).encode("utf-8")

    streaming = req_data.get("stream", False)
    messages = req_data.get("messages", [])
    messages_str = json.dumps(messages)
    headers = dict(request.headers)
    headers.pop("Host", None)
    headers["Content-Length"] = str(len(body))

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{BACKEND_URL}/v1/chat/completions",
                data=body,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                if streaming:
                    return await _handle_streaming_response(
                        resp,
                        start_time,
                        messages_str,
                        request,
                    )
                return await _handle_non_streaming_response(
                    resp,
                    start_time,
                    messages_str,
                    request,
                    force_json,
                )
    except aiohttp.ClientConnectorError as exc:
        _log_request(request.path, "?", 0, 0, start_time)
        return aiohttp.web.json_response(
            {"error": f"Backend unreachable: {exc}"},
            status=503,
        )
    except asyncio.TimeoutError:
        _log_request(request.path, "?", 0, 0, start_time)
        return aiohttp.web.json_response({"error": "Backend timeout"}, status=504)
    except Exception as exc:
        _log_request(request.path, "?", 0, 0, start_time)
        return aiohttp.web.json_response(
            {"error": f"Proxy error: {exc}"},
            status=500,
        )


async def _handle_streaming_response(resp, start_time, messages_str, request):
    stripper = ThinkStripper()
    buffered_lines = []
    completion_id = None
    model = None
    created = None
    usage_seen = False
    prompt_tokens = 0
    completion_tokens = 0
    raw_content_chars = 0

    try:
        async for line in resp.content:
            line_str = line.decode("utf-8", errors="ignore").rstrip("\n")
            if not line_str:
                continue

            if line_str.startswith("data: "):
                data_part = line_str[6:]
                if data_part == "[DONE]":
                    buffered_lines.append(line_str)
                    break

                try:
                    chunk = json.loads(data_part)
                except json.JSONDecodeError:
                    buffered_lines.append(line_str)
                    continue

                if completion_id is None:
                    completion_id = chunk.get("id")
                    model = chunk.get("model")
                    created = chunk.get("created")
                if "usage" in chunk:
                    usage_seen = True
                    usage = chunk["usage"]
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                if "choices" in chunk and isinstance(chunk["choices"], list):
                    for choice in chunk["choices"]:
                        if "delta" in choice and "content" in choice["delta"]:
                            original = choice["delta"]["content"]
                            stripped = stripper.feed(original)
                            raw_content_chars += len(stripped)
                            choice["delta"]["content"] = stripped

                buffered_lines.append(f"data: {json.dumps(chunk)}")
                continue

            buffered_lines.append(line_str)
    except Exception:
        pass

    flushed = stripper.flush()
    raw_content_chars += len(flushed)
    if not usage_seen:
        prompt_tokens = len(messages_str) // 4
        completion_tokens = raw_content_chars // 4

    response_body = []
    for line in buffered_lines:
        response_body.append(line.encode("utf-8") + b"\n")

    if completion_id:
        usage_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created or int(time.time()),
            "model": model or "qwen3.5-9b-mlx",
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        response_body.append(f"data: {json.dumps(usage_chunk)}\n".encode("utf-8"))

    response_body.append(b"data: [DONE]\n")
    _log_request(request.path, "streaming", prompt_tokens, completion_tokens, start_time)
    return aiohttp.web.Response(
        body=b"".join(response_body),
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


async def _handle_non_streaming_response(
    resp,
    start_time,
    messages_str,
    request,
    force_json,
):
    try:
        body = await resp.read()
        resp_data = json.loads(body)
    except Exception as exc:
        return aiohttp.web.json_response(
            {"error": f"Failed to parse backend response: {exc}"},
            status=resp.status,
        )

    stripper = ThinkStripper()
    if "choices" in resp_data and isinstance(resp_data["choices"], list):
        for choice in resp_data["choices"]:
            if "message" in choice and "content" in choice["message"]:
                original = choice["message"]["content"]
                stripped = stripper.feed(original)
                stripped += stripper.flush()
                if force_json:
                    stripped = _clean_json_only_content(stripped)
                choice["message"]["content"] = stripped

    if "usage" not in resp_data:
        content_length = 0
        if "choices" in resp_data and len(resp_data["choices"]) > 0:
            content_length = len(
                resp_data["choices"][0].get("message", {}).get("content", "")
            )
        resp_data["usage"] = {
            "prompt_tokens": len(messages_str) // 4,
            "completion_tokens": content_length // 4,
            "total_tokens": (len(messages_str) + content_length) // 4,
        }

    usage = resp_data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    _log_request(
        request.path,
        "non-streaming",
        prompt_tokens,
        completion_tokens,
        start_time,
    )
    return aiohttp.web.json_response(resp_data, status=resp.status)


async def handle_passthrough(request):
    start_time = time.monotonic()
    try:
        body = await request.read()
    except Exception:
        body = b""

    headers = dict(request.headers)
    headers.pop("Host", None)
    if body:
        headers["Content-Length"] = str(len(body))

    method = request.method
    path = request.path
    if request.query_string:
        path += f"?{request.query_string}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method,
                f"{BACKEND_URL}{path}",
                data=body if body else None,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                response_body = await resp.read()
                response_headers = {}
                for key, value in resp.headers.items():
                    if key.lower() not in ("content-length", "transfer-encoding"):
                        response_headers[key] = value
                _log_request(path, "passthrough", 0, 0, start_time)
                return aiohttp.web.Response(
                    body=response_body,
                    status=resp.status,
                    headers=response_headers,
                )
    except aiohttp.ClientConnectorError:
        _log_request(path, "passthrough", 0, 0, start_time)
        return aiohttp.web.json_response({"error": "Backend unreachable"}, status=503)
    except asyncio.TimeoutError:
        _log_request(path, "passthrough", 0, 0, start_time)
        return aiohttp.web.json_response({"error": "Backend timeout"}, status=504)
    except Exception as exc:
        _log_request(path, "passthrough", 0, 0, start_time)
        return aiohttp.web.json_response(
            {"error": f"Proxy error: {exc}"},
            status=500,
        )


def _log_request(path, mode, prompt_tokens, completion_tokens, start_time):
    elapsed_ms = (time.monotonic() - start_time) * 1000
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[{ts}] {path} | mode={mode} | prompt_tokens={prompt_tokens} | "
        f"completion_tokens={completion_tokens} | elapsed_ms={elapsed_ms:.2f}",
        flush=True,
    )


async def main():
    app = aiohttp.web.Application()
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_route("*", "/{path_info:.*}", handle_passthrough)
    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, "127.0.0.1", LISTEN_PORT)
    await site.start()
    print(
        f"mlx-token-proxy listening on port {LISTEN_PORT}, forwarding to "
        f"{BACKEND_URL}",
        flush=True,
    )
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("\nShutting down proxy.", flush=True)
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
