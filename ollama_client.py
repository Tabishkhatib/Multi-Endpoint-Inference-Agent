"""
ollama_client.py

Talks to ONE Ollama endpoint at a time. Streams tokens back one by one,
timestamping each one as it arrives, so the caller (agent.py) can watch
the pace of generation in real time and detect stalls.

This file does NOT decide anything. It just reports what's happening on
the wire, honestly, including when the connection dies.
"""

import time
import json
import requests # type: ignore

# Port -> model mapping. This mapping is enforced HERE, in our own code.
# Ollama itself doesn't know "port 11434 = llama3.2:1b" — it's our convention.
ENDPOINTS = {
    "fast": {
        "host": "127.0.0.1",
        "port": 11437,
        "model": "llama3.2:1b",
        "category": "simple",
    },
    "reasoning": {
        "host": "127.0.0.1",
        "port": 11435,
        "model": "qwen2.5:3b",
        "category": "reasoning",
    },
    "code": {
        "host": "127.0.0.1",
        "port": 11436,
        "model": "qwen2.5-coder:1.5b",
        "category": "code",
    },
}


class ConnectionDropped(Exception):
    """Raised when the stream dies mid-generation (hard failure)."""
    pass


def stream_generate(endpoint_name: str, prompt: str, connect_timeout=5, read_timeout=8):
    """
    Streams a generation from the given endpoint.

    Yields a dict for every token chunk:
        {
            "token": str,              # the text piece
            "token_index": int,        # 1-based count of tokens so far
            "gap_ms": float,           # ms since the previous token (or since request start, for the first token)
            "elapsed_total_ms": float, # ms since the request was sent
        }

    On successful completion, the generator simply ends (StopIteration).
    On a dead connection / timeout, raises ConnectionDropped.
    """
    if endpoint_name not in ENDPOINTS:
        raise ValueError(f"Unknown endpoint: {endpoint_name}")

    cfg = ENDPOINTS[endpoint_name]
    url = f"http://{cfg['host']}:{cfg['port']}/api/generate"
    payload = {
        "model": cfg["model"],
        "prompt": prompt,
        "stream": True,
    }

    request_start = time.monotonic()
    last_token_time = request_start
    token_index = 0

    try:
        response = requests.post(
            url,
            json=payload,
            stream=True,
            timeout=(connect_timeout, read_timeout),
        )
        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            now = time.monotonic()
            gap_ms = (now - last_token_time) * 1000
            elapsed_total_ms = (now - request_start) * 1000
            last_token_time = now

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip malformed line, don't crash the whole stream

            token_text = chunk.get("response", "")
            token_index += 1

            yield {
                "token": token_text,
                "token_index": token_index,
                "gap_ms": gap_ms,
                "elapsed_total_ms": elapsed_total_ms,
                "is_first_token": token_index == 1,
            }

            if chunk.get("done"):
                return  # clean finish

    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        # This is the "hard failure" case — connection actually died or hung
        # past our read_timeout. The caller (Decision Engine) treats this
        # differently from a mere slow-but-alive stall.
        raise ConnectionDropped(f"{endpoint_name} ({cfg['model']}): {e}") from e


def quick_health_check(endpoint_name: str, timeout=3) -> bool:
    """Fast, non-streaming check: is this endpoint even reachable right now?
    Used by the Router before picking a backup endpoint mid-decision."""
    cfg = ENDPOINTS[endpoint_name]
    url = f"http://{cfg['host']}:{cfg['port']}/api/tags"
    try:
        r = requests.get(url, timeout=timeout)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


if __name__ == "__main__":
    # Manual smoke test: run `python ollama_client.py` to sanity check
    # streaming against the fast endpoint.
    print("Streaming from 'fast' endpoint...\n")
    for chunk in stream_generate("fast", "Say hello in one sentence."):
        print(f"[token {chunk['token_index']:>3}] gap={chunk['gap_ms']:.1f}ms  '{chunk['token']}'")
