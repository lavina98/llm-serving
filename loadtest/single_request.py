"""Day 1: send one streaming request to an OpenAI-compatible server and time it.

Usage:
    python loadtest/single_request.py
    python loadtest/single_request.py --prompt "Explain KV cache" --max-tokens 512 --repeats 5
"""
import argparse
import json
import statistics
import time

import httpx

DEFAULT_URL = "http://localhost:8000/v1/chat/completions"
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def build_payload(model: str, prompt: str, max_tokens: int) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        # Ask the server for a final chunk with exact token counts.
        "stream_options": {"include_usage": True},
        # vLLM extension: keep generating past the end-of-sequence token,
        # so every run produces exactly max_tokens and runs are comparable.
        "ignore_eos": True,
    }


def parse_sse_line(line: str) -> dict | None:
    """Return the JSON payload of one server-sent-events line.

    Return None for blank lines and for the final 'data: [DONE]' marker.
    """
    # TODO(you): lines look like 'data: {...json...}'.
    #   1. Skip empty lines.
    #   2. Strip the 'data: ' prefix.
    #   3. Return None for '[DONE]'.
    #   4. json.loads the rest.
    raise NotImplementedError


def run_once(client: httpx.Client, url: str, payload: dict) -> dict:
    t_start = time.perf_counter()
    t_first = None          # arrival time of the first chunk with text
    n_content_chunks = 0
    text_parts = []
    usage = None

    with client.stream("POST", url, json=payload) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            event = parse_sse_line(line)
            if event is None:
                continue
            # TODO(you): handle one event.
            #   - If event.get("usage") is set, save it in `usage`
            #     (it arrives in the last chunk, which has no choices).
            #   - For each choice in event["choices"], read choice["delta"].get("content").
            #   - When there is content: set t_first if this is the first one,
            #     count the chunk, and append the text.
            raise NotImplementedError

    t_end = time.perf_counter()

    # TODO(you): compute and return these metrics.
    #   ttft_ms        send -> first content chunk
    #   e2e_ms         send -> last byte
    #   prompt_tokens  usage["prompt_tokens"]  (note: larger than your prompt, why?)
    #   output_tokens  usage["completion_tokens"]
    #   tpot_ms        time per output token after the first:
    #                  (t_end - t_first) / (output_tokens - 1), in ms
    #   decode_tok_s   1000 / tpot_ms
    #   n_chunks       n_content_chunks (compare it with output_tokens)
    raise NotImplementedError


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt", default="Explain what a GPU does, in detail.")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()

    payload = build_payload(args.model, args.prompt, args.max_tokens)
    with httpx.Client(timeout=300) as client:
        if not args.no_warmup:
            warm = run_once(client, args.url, payload)
            print(f"warmup: ttft {warm['ttft_ms']:.1f} ms, {warm['decode_tok_s']:.1f} tok/s")

        runs = [run_once(client, args.url, payload) for _ in range(args.repeats)]

    print(f"\nprompt_tokens={runs[0]['prompt_tokens']} output_tokens={runs[0]['output_tokens']} "
          f"chunks={runs[0]['n_chunks']}")
    for key in ("ttft_ms", "tpot_ms", "decode_tok_s", "e2e_ms"):
        vals = [r[key] for r in runs]
        print(f"{key:>13}: median {statistics.median(vals):8.1f}   "
              f"min {min(vals):8.1f}   max {max(vals):8.1f}")


if __name__ == "__main__":
    main()
