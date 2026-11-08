#!/usr/bin/env python3
"""Ask Dory for one harness action per Jotnar task.

This is behavior-level teacher data: the exact gate decides whether the union
of the teacher's actions is adopted.
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

ACTIONS = [
    "add compose[sum4]",
    "depth + 1",
    "refine off",
    "tags off",
    "add family select_max",
    "refinements + 1",
    "add family select_min",
    "add family saxpy_dup",
    "add family affine_dup",
    "add family matmul2_dup",
]


def prompt(record, program, effects):
    task = record["task"]
    pairs = ", ".join(f"{n}={v}" for n, v in zip(FEATURE_NAMES, task))
    actions = "\n".join(f"{i}: {name}" for i, name in enumerate(ACTIONS))
    listing = ""
    if program:
        listing = (
            f"Start listing:\n{program.get('start_text', '')}\n"
            f"Target listing:\n{program.get('target_text', '')}\n"
            f"Example inputs: {json.dumps(program.get('example_inputs', []))}\n"
        )
    effect_text = ""
    if effects:
        lines = []
        for effect in effects.get("actions", []):
            if not effect.get("applicable", False):
                lines.append(f"{effect['index']} {effect['label']}: not applicable")
            else:
                lines.append(
                    f"{effect['index']} {effect['label']}: {effect['outcome']}"
                    f" ({effect.get('steps', 0)} steps; {effect.get('note', '')})"
                )
        effect_text = "Action effects on the examples:\n" + "\n".join(lines) + "\n"
    return (
        "You are the harness optimizer for a Jotnar program-repair agent.\n"
        f"Case: {record.get('case', 'unknown')}\n"
        f"Base outcome: {effects.get('base_outcome', 'unknown') if effects else 'unknown'}\n"
        f"Task features: {pairs}.\n"
        f"{listing}"
        f"{effect_text}"
        "The bounded harness actions are:\n"
        f"{actions}\n"
        "Choose the single action most likely to turn this case from a wrong "
        "refusal into an exact solve. Return only JSON: {\"action\": N}."
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
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["choices"][0]["message"].get("content", "")


def parse_action(text):
    try:
        obj = json.loads(text)
        value = obj.get("action", obj)
        if isinstance(value, str):
            value = ACTIONS.index(value)
        value = int(value)
        if 0 <= value < len(ACTIONS):
            return value
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    match = re.search(r'"?action"?\s*[:=]\s*"?(\d+)', text)
    if match:
        value = int(match.group(1))
        if 0 <= value < len(ACTIONS):
            return value
    for index, name in enumerate(ACTIONS):
        if name in text:
            return index
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--programs", default=None)
    parser.add_argument("--effects", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--model", default="dory")
    args = parser.parse_args()

    records = [json.loads(line) for line in open(args.tasks, encoding="utf-8") if line.strip()]
    programs = {}
    if args.programs:
        programs = {
            row["id"]: row
            for row in (json.loads(line) for line in open(args.programs, encoding="utf-8") if line.strip())
        }
    effects = {}
    if args.effects:
        effects = {
            row["case"]: row
            for row in (json.loads(line) for line in open(args.effects, encoding="utf-8") if line.strip())
        }
    written = failed = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for index, record in enumerate(records, 1):
            response = ask(
                args.url,
                args.model,
                prompt(
                    record,
                    programs.get(record.get("case", "")),
                    effects.get(record.get("case", "")),
                ),
            )
            action = parse_action(response)
            if action is None:
                failed += 1
                print(f"[{index}/{len(records)}] parse failed: {response[:160]!r}")
                continue
            row = {
                "case": record.get("case", ""),
                "task": record["task"],
                "action": action,
                "split": record.get("split", "stream"),
                "source": "dory",
            }
            out.write(json.dumps(row) + "\n")
            written += 1
            print(f"[{index}/{len(records)}] {row['case']} -> {action}: {ACTIONS[action]}")
    print(f"wrote {written} rows to {args.out} ({failed} parse failures)")


if __name__ == "__main__":
    main()
