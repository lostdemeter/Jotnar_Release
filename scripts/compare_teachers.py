#!/usr/bin/env python3
"""Compare teacher latent JSONL files (loop counts, teachers, named baseline)."""

import argparse
import json
from collections import defaultdict

CHANNELS = [
    "start_outputs", "start_ops", "target_ops", "depth_delta",
    "wrong_refusal", "start_is_four", "start_is_one", "has_sum4",
    "target_outputs_1",
]


def load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", action="append", required=True,
                        help="label:path (repeatable)")
    args = parser.parse_args()

    named = {}
    sources = {}
    for spec in args.teacher:
        label, path = spec.split(":", 1)
        rows = load(path)
        sources[label] = rows
        for row in rows:
            named.setdefault(tuple(row["task"]), row["latent"])

    print("channel means")
    header = f"{'channel':<18}" + "".join(f"{label:>10}" for label in sources)
    print(header)
    for index, name in enumerate(CHANNELS):
        values = []
        for label, rows in sources.items():
            mean = sum(row["latent"][index] for row in rows) / max(1, len(rows))
            values.append(f"{mean:10.3f}")
        print(f"{name:<18}" + "".join(values))

    print("\npairwise mean absolute difference")
    labels = list(sources)
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            left = {tuple(row["task"]): row["latent"] for row in sources[a]}
            right = {tuple(row["task"]): row["latent"] for row in sources[b]}
            keys = sorted(set(left) & set(right))
            if not keys:
                continue
            diff = sum(
                abs(left[key][c] - right[key][c])
                for key in keys for c in range(len(CHANNELS))
            ) / (len(keys) * len(CHANNELS))
            changed = sum(1 for key in keys if left[key] != right[key])
            print(f"{a} vs {b}: mean|delta|={diff:.4f}, changed={changed}/{len(keys)}")

    print("\nnon-zero channels by teacher")
    for label, rows in sources.items():
        counts = [
            sum(1 for row in rows if abs(row["latent"][c]) > 1e-9)
            for c in range(len(CHANNELS))
        ]
        print(f"{label}: " + ", ".join(f"{CHANNELS[c]}={counts[c]}" for c in range(len(CHANNELS))))


if __name__ == "__main__":
    main()
