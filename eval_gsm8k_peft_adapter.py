import argparse
import json
import os
import re

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def format_prompt(question: str) -> str:
    return f"Question:\n{question.strip()}\n\nAnswer:\n"


def extract_gsm8k_answer(text: str) -> str:
    if "####" in text:
        text = text.split("####")[-1]
    matches = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if not matches:
        return text.strip()
    return matches[-1].replace(",", "").strip()


def load_split(dataset_name: str, dataset_config: str, split: str):
    if os.path.isdir(dataset_name):
        try:
            dataset = load_from_disk(dataset_name)
            if isinstance(dataset, DatasetDict):
                return dataset[split]
            return dataset
        except Exception:
            data_files = {}
            for name in ("train", "validation", "test"):
                for extension in ("jsonl", "json", "parquet", "csv"):
                    path = os.path.join(dataset_name, f"{name}.{extension}")
                    if os.path.exists(path):
                        data_files[name] = path
            if not data_files:
                raise
            extension = os.path.splitext(next(iter(data_files.values())))[1].lstrip(".")
            loader = "json" if extension in {"json", "jsonl"} else extension
            return load_dataset(loader, data_files=data_files, split=split)
    if dataset_config:
        return load_dataset(dataset_name, dataset_config, split=split)
    return load_dataset(dataset_name, split=split)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset_config", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument("--question_column", default="input")
    parser.add_argument("--answer_column", default="output")
    parser.add_argument("--metrics_file", required=True)
    parser.add_argument("--predictions_file", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.to(device)
    model.eval()

    dataset = load_split(args.dataset, args.dataset_config, args.split)
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    os.makedirs(os.path.dirname(args.metrics_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.predictions_file), exist_ok=True)

    correct = 0
    total = 0
    with open(args.predictions_file, "w", encoding="utf-8") as out:
        for start in tqdm(range(0, len(dataset), args.batch_size), desc="GSM8K eval"):
            batch = dataset[start : start + args.batch_size]
            questions = batch[args.question_column]
            answers = batch[args.answer_column]
            prompts = [format_prompt(question) for question in questions]
            inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
            prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            for index, output_ids in enumerate(generated):
                new_tokens = output_ids[int(prompt_lengths[index]) :]
                prediction_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                pred_answer = extract_gsm8k_answer(prediction_text)
                gold_answer = extract_gsm8k_answer(str(answers[index]))
                is_correct = pred_answer == gold_answer
                correct += int(is_correct)
                total += 1
                out.write(
                    json.dumps(
                        {
                            "question": questions[index],
                            "prediction": prediction_text,
                            "pred_answer": pred_answer,
                            "gold_answer": gold_answer,
                            "correct": is_correct,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                out.flush()

    metrics = {"accuracy": correct / total if total else 0.0, "correct": correct, "total": total}
    with open(args.metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
