import argparse
import json
import os
import re
from argparse import Namespace

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.lora_modules import prepare_lora_for_inference, switch_to_lora
from common.lora_modules.gora import LinearWithGoRA
from common.utils import load_ckpt


def normalize_number(text):
    if text is None:
        return None
    text = str(text).replace(",", "").strip()
    match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if match:
        return match.group(1)
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        nums = re.findall(r"-?\d+(?:\.\d+)?", boxed[-1].replace(",", ""))
        if nums:
            return nums[-1]
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def load_rows(path, read_num=None):
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".jsonl"):
        df = pd.read_json(path, lines=True)
    elif path.endswith(".json"):
        df = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported dataset format: {path}")

    df = df.rename(
        columns={
            "question": "input",
            "answer": "output",
            "problem": "input",
        }
    )
    if "input" not in df.columns or "output" not in df.columns:
        raise ValueError(f"Dataset must contain input/output or question/answer columns, got {list(df.columns)}")
    if read_num:
        df = df.head(read_num)
    return df[["input", "output"]].to_dict("records")


def make_prompt(tokenizer, question):
    if not getattr(tokenizer, "chat_template", None):
        return (
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{question}"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_training_args(adapter_dir):
    with open(os.path.join(adapter_dir, "config.json"), "r", encoding="utf-8") as f:
        args = Namespace(**json.load(f))
    args.global_rank = getattr(args, "global_rank", 0)
    args.local_rank = getattr(args, "local_rank", 0)
    args.device = getattr(args, "device", "cuda")
    args.gora_adapter_type = getattr(args, "gora_adapter_type", "lora")
    return args


def prepare_adapter(model, train_args):
    if getattr(train_args, "use_gora", False):
        switch_to_lora(model, train_args)
        rank_path = os.path.join(train_args.output_path, train_args.experiment_name, "rank.json")
        with open(rank_path, "r", encoding="utf-8") as f:
            rank_config = json.load(f)
        for name, module in model.model.named_modules():
            if isinstance(module, LinearWithGoRA):
                rank_name = name if name in rank_config else f"model.{name}"
                if rank_name not in rank_config:
                    raise KeyError(f"Missing GoRA rank for module {name}")
                module.init_method = "vanilla"
                module.dynamic_init(train_args.lora_rank, rank_config[rank_name])
    else:
        prepare_lora_for_inference(model, train_args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--save_file", required=True)
    parser.add_argument("--metrics_file", required=True)
    parser.add_argument("--ckpt_name", default="final.ckpt")
    parser.add_argument("--read_num", type=int, default=None)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)

    train_args = load_training_args(args.adapter_dir)
    prepare_adapter(model, train_args)
    load_ckpt(model=model, partial_ckpt_path=os.path.join(args.adapter_dir, args.ckpt_name), rank=0)
    model.to(device)
    model.eval()

    rows = load_rows(args.dataset, args.read_num)
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    rows = rows[args.shard_index :: args.num_shards]
    os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
    correct = 0
    missing = 0

    with open(args.save_file, "w", encoding="utf-8") as out:
        for start in tqdm(range(0, len(rows), args.batch_size), desc="gsm8k-gora"):
            batch = rows[start : start + args.batch_size]
            prompts = [make_prompt(tokenizer, row["input"]) for row in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            prompt_len = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            predictions = tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)
            for row, pred in zip(batch, predictions):
                pred_answer = normalize_number(pred)
                gold_answer = normalize_number(row["output"])
                is_correct = pred_answer is not None and gold_answer is not None and pred_answer == gold_answer
                correct += int(is_correct)
                missing += int(pred_answer is None)
                out.write(
                    json.dumps(
                        {
                            "question": row["input"],
                            "prediction": pred,
                            "pred_answer": pred_answer,
                            "gold_answer": gold_answer,
                            "correct": is_correct,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                out.flush()

    metrics = {
        "accuracy": correct / len(rows) if rows else 0.0,
        "correct": correct,
        "total": len(rows),
        "missing_pred": missing,
    }
    with open(args.metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
