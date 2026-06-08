#!/usr/bin/env python
import argparse
import json
import os
from collections import Counter

from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare CodeFeedback in the local jsonl format used by GoRA-style SFT datasets."
    )
    parser.add_argument("--dataset_name", type=str, default="m-a-p/CodeFeedback-Filtered-Instruction")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_column", type=str, default="query")
    parser.add_argument("--output_column", type=str, default="answer")
    parser.add_argument("--lang", type=str, default=None, help="Optional language filter, e.g. python.")
    parser.add_argument("--max_answer_chars", type=int, default=None)
    parser.add_argument("--max_query_chars", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_dataset(args.dataset_name, split=args.split)
    if args.max_samples is not None and args.max_samples < len(dataset):
        dataset = dataset.shuffle(seed=args.seed).select(range(args.max_samples))

    output_path = os.path.join(args.output_dir, "train.jsonl")
    stats = Counter()
    resource_stats = Counter()
    lang_stats = Counter()

    with open(output_path, "w", encoding="utf-8") as f:
        for index, example in enumerate(dataset):
            query = str(example.get(args.input_column, "") or "")
            answer = str(example.get(args.output_column, "") or "")
            resource = str(example.get("resource", "") or "")
            lang = str(example.get("lang", "") or "")

            stats["seen"] += 1
            if args.lang is not None and lang.lower() != args.lang.lower():
                stats["skipped_lang"] += 1
                continue
            if not query.strip() or not answer.strip():
                stats["skipped_empty"] += 1
                continue
            if args.max_query_chars is not None and len(query) > args.max_query_chars:
                stats["skipped_long_query"] += 1
                continue
            if args.max_answer_chars is not None and len(answer) > args.max_answer_chars:
                stats["skipped_long_answer"] += 1
                continue

            row = {
                "input": query,
                "output": answer,
                "query": query,
                "answer": answer,
                "resource": resource,
                "lang": lang,
                "source_index": index,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["written"] += 1
            resource_stats[resource] += 1
            lang_stats[lang] += 1

    summary = {
        "dataset_name": args.dataset_name,
        "split": args.split,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "output_path": output_path,
        "stats": dict(stats),
        "resource_counts": dict(resource_stats.most_common()),
        "lang_counts": dict(lang_stats.most_common()),
        "format": {
            "gora_input_field": "input",
            "gora_output_field": "output",
            "gslora_question_column": "query",
            "gslora_answer_column": "answer",
        },
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
