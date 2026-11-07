#!/usr/bin/env python3
"""Generate Jotnar teacher rows from a local Ollama model.

Input is the JSONL from `jotnar-agent gen-teacher` (task vector + named
latent). Output is the same schema with the model's 9-channel latent and
`source` set to the model name. This is the non-Dory teacher path; the same
schema is used by Dory when it is available.
"""

import argparse
import json
import urllib.request


FEATURE_NAMES = [
    "start_inputs",
    "start_outputs",
    "start_ops",
    "target_inputs",
    "target_outputs",
    "target_ops",
    "depth_delta",
    "output_delta",
    "input_delta",
    "expect_solve",
    "wrong_refusal",
    "wrong_solve",
    "steps",
    "refined",
    "start_outputs_1",
    "start_outputs_4",
    "target_outputs_1",
    "has_sum4",
    "refinements",
    "tags_count",
]

LATENT_NAMES = [
    "start_outputs",
    "start_ops",
    "target_ops",
    "depth_delta",
    "wrong_refusal",
    "start_is_four",
    "start_is_one",
    "has_sum4",
    "target_outputs_1",
]


def prompt(task):
    pairs = ", ".join(f"{name}={value}" for name, value in zip(FEATURE_NAMES, task))
    channels = ", ".join(f"{i}={name}" for i, name in enumerate(LATENT_NAMES))
    return (
        "You are the task encoder for a Jotnar program-repair agent.\n"
        f"Task features: {pairs}.\n"
        f"Produce a 9-dimensional latent vector. Channels: {channels}.\n"
        "Return only JSON with this exact shape: "
        '{"latent":[n0,n1,n2,n3,n4,n5,n6,n7,n8]}.\n'
        "Use small integers or decimals. Do not explain."
    )


def ask(model, url, text):
    body = json.dumps(
        {
            "model": model,
            "prompt": text,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": 128},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload.get("response", "")


def parse_latent(text):
    try:
        obj = json.loads(text)
        values = obj.get("latent", obj)
        if isinstance(values, dict):
            values = [values[name] for name in LATENT_NAMES]
        if isinstance(values, list) and len(values) == len(LATENT_NAMES):
            return [max(-128.0, min(128.0, float(value))) for value in values]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="qwen2:7b")
    parser.add_argument("--url", default="http://localhost:11434/api/generate")
    args = parser.parse_args()

    records = []
    with open(args.tasks, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))

    written = 0
    failed = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for index, record in enumerate(records, 1):
            task = record["task"]
            response = ask(args.model, args.url, prompt(task))
            latent = parse_latent(response)
            if latent is None:
                failed += 1
                print(f"[{index}/{len(records)}] parse failed: {response[:120]!r}")
                continue
            row = {
                "case": record.get("case", ""),
                "task": task,
                "latent": latent,
                "split": record.get("split", "stream"),
                "source": args.model,
            }
            out.write(json.dumps(row) + "\n")
            written += 1
            print(f"[{index}/{len(records)}] {record.get('split', 'stream')} -> {latent}")

    print(f"wrote {written} rows to {args.out} ({failed} parse failures)")


if __name__ == "__main__":
    main()
