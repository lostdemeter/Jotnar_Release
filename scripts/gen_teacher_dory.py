#!/usr/bin/env python3
"""Generate Jotnar teacher rows from a local Dory llama-server.

Dory is served by the Dory-enabled llama.cpp build:

  llama-server -m dory-q5_k_m.gguf --alias dory --port 8080 \
    --jinja --reasoning on \
    --override-kv "dory.num_loops=int:2,dory.recurrent_kv_cache_mode=str:per_loop"

Input is the JSONL from `jotnar-agent gen-teacher`; output is the same schema
with the Dory 9-channel latent and `source="dory"`.
"""

import argparse
import json
import re
import urllib.request


FEATURE_NAMES = [
    "start_inputs", "start_outputs", "start_ops", "target_inputs",
    "target_outputs", "target_ops", "depth_delta", "output_delta",
    "input_delta", "expect_solve", "wrong_refusal", "wrong_solve",
    "steps", "refined", "start_outputs_1", "start_outputs_4",
    "target_outputs_1", "has_sum4", "refinements", "tags_count",
]
LATENT_NAMES = [
    "start_outputs", "start_ops", "target_ops", "depth_delta",
    "wrong_refusal", "start_is_four", "start_is_one", "has_sum4",
    "target_outputs_1",
]


def prompt(task):
    pairs = ", ".join(f"{n}={v}" for n, v in zip(FEATURE_NAMES, task))
    channels = ", ".join(f"{i}={n}" for i, n in enumerate(LATENT_NAMES))
    return (
        "You are the task encoder for a Jotnar program-repair agent.\n"
        f"Task features: {pairs}.\n"
        f"Produce a 9-dimensional latent vector. Channels: {channels}.\n"
        "Return only JSON with this exact shape: "
        '{"latent":[n0,n1,n2,n3,n4,n5,n6,n7,n8]}.\n'
        "Use small integers or decimals. Do not explain."
    )


def ask(url, model, text):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "temperature": 0.0,
        "max_tokens": 256,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["choices"][0]["message"].get("content", "")


def parse_latent(text):
    try:
        obj = json.loads(text)
        values = obj.get("latent", obj)
        if isinstance(values, dict):
            values = [values[name] for name in LATENT_NAMES]
        if isinstance(values, list) and len(values) == 9:
            return [max(-128.0, min(128.0, float(v))) for v in values]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    # Reasoning models sometimes wrap the JSON in prose; take the first array.
    match = re.search(r"\[([^\]]+)\]", text)
    if match:
        try:
            values = [float(x) for x in match.group(1).split(",")]
            if len(values) == 9:
                return [max(-128.0, min(128.0, v)) for v in values]
        except ValueError:
            pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--model", default="dory")
    args = parser.parse_args()

    records = [json.loads(line) for line in open(args.tasks, encoding="utf-8") if line.strip()]
    written = failed = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for index, record in enumerate(records, 1):
            response = ask(args.url, args.model, prompt(record["task"]))
            latent = parse_latent(response)
            if latent is None:
                failed += 1
                print(f"[{index}/{len(records)}] parse failed: {response[:160]!r}")
                continue
            row = {
                "case": record.get("case", ""),
                "task": record["task"],
                "latent": latent,
                "split": record.get("split", "stream"),
                "source": "dory",
            }
            out.write(json.dumps(row) + "\n")
            written += 1
            print(f"[{index}/{len(records)}] {row['split']} -> {latent}")
    print(f"wrote {written} rows to {args.out} ({failed} parse failures)")


if __name__ == "__main__":
    main()
