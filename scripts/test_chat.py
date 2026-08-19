#!/usr/bin/env python3
"""Smoke test for the deployed endpoint: streamed chat (thinking mode) and a
tool-call round trip (instruct-style sampling).

    python scripts/test_chat.py --base-url http://IP:PORT/v1

Requires: pip install openai
"""
import argparse
import json
import time

from openai import OpenAI


def stream_chat(client: OpenAI, model: str) -> None:
    print("=== streamed chat (thinking mode: temp=1.0 top_p=0.95 top_k=20) ===")
    start = time.perf_counter()
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "In two sentences, why does FP8 "
                   "quantization let a 27B model fit on a 48 GB GPU?"}],
        temperature=1.0,
        top_p=0.95,
        extra_body={"top_k": 20},
        max_tokens=1024,
        stream=True,
    )
    first_token = None
    thinking = False
    for chunk in stream:
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            if not thinking:
                print("[thinking] ", end="", flush=True)
                thinking = True
            continue  # don't print the reasoning itself, just note it happened
        if delta.content:
            if first_token is None:
                first_token = time.perf_counter() - start
            print(delta.content, end="", flush=True)
    print(f"\n[ttft {first_token:.2f}s, total {time.perf_counter() - start:.2f}s]\n")


def tool_call(client: OpenAI, model: str) -> None:
    print("=== tool call (instruct: temp=0.7 top_p=0.8 presence_penalty=1.5) ===")
    tools = [{
        "type": "function",
        "function": {
            "name": "get_gpu_price",
            "description": "Get the current hourly rental price for a GPU model",
            "parameters": {
                "type": "object",
                "properties": {"gpu": {"type": "string"}},
                "required": ["gpu"],
            },
        },
    }]
    messages = [{"role": "user",
                 "content": "How much does an RTX 6000 Ada cost per hour to rent?"}]
    resp = client.chat.completions.create(
        model=model, messages=messages, tools=tools,
        temperature=0.7, top_p=0.8, presence_penalty=1.5, max_tokens=1024,
    )
    call = resp.choices[0].message.tool_calls[0]
    print(f"model called {call.function.name}({call.function.arguments})")

    messages.append(resp.choices[0].message)
    messages.append({"role": "tool", "tool_call_id": call.id,
                     "content": json.dumps({"gpu": "RTX 6000 Ada",
                                            "usd_per_hour": 0.65})})
    final = client.chat.completions.create(
        model=model, messages=messages, tools=tools,
        temperature=0.7, top_p=0.8, presence_penalty=1.5, max_tokens=1024,
    )
    print(f"final answer: {final.choices[0].message.content}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True,
                        help="e.g. http://IP:PORT/v1")
    parser.add_argument("--model", default="qwen3.8-27b")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="unused")
    served = [m.id for m in client.models.list().data]
    print(f"models on server: {served}")
    stream_chat(client, args.model)
    tool_call(client, args.model)
    print("smoke test passed")


if __name__ == "__main__":
    main()
