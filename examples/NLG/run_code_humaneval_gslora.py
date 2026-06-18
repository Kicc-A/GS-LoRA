import argparse
import json
import multiprocessing as mp
import os
import re
import tempfile
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import run_math_gslora as base
from gs_lora import GSLoraConfig, prepare_gslora_model

CODE_PROMPT_TEMPLATE = """<|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

HUMANEVAL_PROMPT_TEMPLATE = """<|start_header_id|>user<|end_header_id|>

Complete the following Python code:
Notes: respond with the entire complete function definition.
Do not add any comments. Be concise in your code.
Use only built-in libraries, assume no additional imports other than those provided.
Use 4 spaces for each level of indentation.

code:
```python
{prompt}
```<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a CausalLM with GS-LoRA on CodeFeedback and evaluate HumanEval pass@1.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_dataset_name", type=str, default="m-a-p/CodeFeedback-Filtered-Instruction")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--train_question_column", type=str, default="query")
    parser.add_argument("--train_answer_column", type=str, default="answer")
    parser.add_argument("--eval_dataset_name", type=str, default="openai/openai_humaneval")
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--target_modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    parser.add_argument("--target_prefix", type=str, default=None)
    parser.add_argument("--base_rank", type=int, default=8)
    parser.add_argument("--tau", type=float, default=0.90)
    parser.add_argument("--r_min", type=int, default=4)
    parser.add_argument("--r_max", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--use_lora_plus", action="store_true", help="Use LoRA+ style higher LR for LoRA B weights")
    parser.add_argument("--lora_plus_scaler", type=float, default=16.0, help="LR multiplier for LoRA B weights when LoRA+ is enabled")
    parser.add_argument("--init_method", choices=["none", "svd_sqrt", "svd_sigma", "svd_a_zero_b", "svd_a_energy_zero_b", "svd_a_energy_small_b"], default="svd_a_zero_b")
    parser.add_argument("--init_scale", type=float, default=1.0)
    parser.add_argument("--init_energy_beta", type=float, default=0.5)
    parser.add_argument("--init_energy_eps", type=float, default=1e-8)
    parser.add_argument("--init_small_b_scale", type=float, default=1e-4)
    parser.add_argument("--scaling_mode", choices=["rank", "sqrt_rank", "avg_rank"], default="rank")
    parser.add_argument("--calibration_steps", type=int, default=64)
    parser.add_argument("--skip_adaptive_rank", action="store_true")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--deepspeed_config", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="gslora-humaneval")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb_dir", type=str, default=None)
    parser.add_argument("--wandb_log_interval", type=int, default=10)
    return parser.parse_args()


def format_code_prompt(instruction: str) -> str:
    return CODE_PROMPT_TEMPLATE.format(instruction=instruction.strip())


def tokenize_sft_example(example, tokenizer, args):
    question = str(example[args.train_question_column])
    answer = str(example[args.train_answer_column])
    prompt = format_code_prompt(question)

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if tokenizer.bos_token_id is not None:
        prompt_ids = [tokenizer.bos_token_id] + prompt_ids
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        answer_ids = answer_ids + [tokenizer.eos_token_id]

    max_answer_length = min(len(answer_ids), args.max_length)
    max_prompt_length = max(args.max_length - max_answer_length, 0)
    if len(prompt_ids) > max_prompt_length:
        prompt_ids = prompt_ids[:max_prompt_length] if max_prompt_length > 0 else []

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    if len(input_ids) > args.max_length:
        input_ids = input_ids[-args.max_length:]
        labels = labels[-args.max_length:]

    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def post_process_completion(text: str) -> str:
    text = text.replace("```python", "```")
    if "```" in text:
        text = text.split("```")[0]
    text = text.replace("\t", "    ").strip()
    if not text:
        return text
    lines = [line.rstrip() for line in text.splitlines()]
    try:
        def_idx = next(i for i, line in enumerate(lines) if re.match(r"\s*def\s+", line))
        lines = lines[def_idx:]
    except StopIteration:
        pass
    return "\n".join(lines).strip() + "\n"


def _run_humaneval_test(code: str, test: str, entry_point: str, queue):
    try:
        namespace = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                exec(code + "\n" + test + f"\ncheck({entry_point})\n", namespace)
            finally:
                os.chdir(old_cwd)
        queue.put((True, ""))
    except BaseException as exc:
        queue.put((False, repr(exc)))


def check_correctness(code: str, test: str, entry_point: str, timeout: float = 3.0):
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    proc = ctx.Process(target=_run_humaneval_test, args=(code, test, entry_point, queue))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.kill()
        proc.join()
        return False, "timeout"
    if queue.empty():
        return False, "no_result"
    return queue.get()


@torch.no_grad()
def evaluate_humaneval(model, tokenizer, eval_dataset, args, device: torch.device) -> Dict[str, object]:
    model.eval()
    records = []
    correct = 0

    for start in tqdm(range(0, len(eval_dataset), args.per_device_eval_batch_size), desc="HumanEval"):
        batch = eval_dataset[start : start + args.per_device_eval_batch_size]
        prompts = [HUMANEVAL_PROMPT_TEMPLATE.format(prompt=prompt) for prompt in batch["prompt"]]
        inputs = tokenizer(prompts, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt").to(device)
        prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        for i, output_ids in enumerate(generated):
            raw = tokenizer.decode(output_ids[int(prompt_lengths[i]) :], skip_special_tokens=True)
            completion = post_process_completion(raw)
            code = batch["prompt"][i] + completion
            passed, error = check_correctness(code, batch["test"][i], batch["entry_point"][i])
            correct += int(passed)
            records.append(
                {
                    "task_id": batch["task_id"][i],
                    "prompt": batch["prompt"][i],
                    "raw_completion": raw,
                    "completion": completion,
                    "passed": passed,
                    "error": error,
                }
            )

    total = len(records)
    return {"pass@1": correct / max(total, 1), "correct": correct, "total": total, "predictions": records}


def main():
    args = parse_args()
    distributed, device = base.setup_distributed(args)
    base.set_seed(args.seed)
    if base.is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
    base.barrier()
    base.init_wandb(args)

    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=dtype, trust_remote_code=args.trust_remote_code).to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    train_dataset = base.load_split(args.train_dataset_name, None, args.train_split)
    eval_dataset = load_dataset(args.eval_dataset_name, split=args.eval_split)
    train_dataset = base.maybe_select(train_dataset, args.max_train_samples, args.seed)
    eval_dataset = base.maybe_select(eval_dataset, args.max_eval_samples, args.seed)

    tokenized_train = train_dataset.map(
        lambda example: tokenize_sft_example(example, tokenizer, args),
        remove_columns=train_dataset.column_names,
    )
    collator = base.SupervisedDataCollator(tokenizer)
    calibration_loader = DataLoader(tokenized_train, batch_size=args.per_device_train_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collator)
    train_sampler = DistributedSampler(tokenized_train, shuffle=True) if distributed else None
    train_loader = DataLoader(tokenized_train, batch_size=args.per_device_train_batch_size, shuffle=train_sampler is None, sampler=train_sampler, num_workers=args.num_workers, collate_fn=collator)

    config = GSLoraConfig(
        target_modules=args.target_modules,
        target_prefix=args.target_prefix,
        base_rank=args.base_rank,
        tau=args.tau,
        r_min=args.r_min,
        r_max=args.r_max,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        calibration_steps=args.calibration_steps,
        adaptive_rank=not args.skip_adaptive_rank,
        init_method=args.init_method,
        init_scale=args.init_scale,
        init_energy_beta=args.init_energy_beta,
        init_energy_eps=args.init_energy_eps,
        init_small_b_scale=args.init_small_b_scale,
        scaling_mode=args.scaling_mode,
    )

    def loss_fn(current_model, batch):
        batch = {key: value.to(device) for key, value in batch.items()}
        return current_model(**batch).loss

    model, gs_report = prepare_gslora_model(model=model, dataloader=calibration_loader, loss_fn=loss_fn, config=config, device=device)
    if base.is_main_process() and hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
        with open(os.path.join(args.output_dir, "gs_lora_report.json"), "w", encoding="utf-8") as f:
            json.dump(gs_report, f, indent=2)
        with open(os.path.join(args.output_dir, "rank_pattern.json"), "w", encoding="utf-8") as f:
            json.dump(gs_report["rank_pattern"], f, indent=2)
        with open(os.path.join(args.output_dir, "rank_stats.json"), "w", encoding="utf-8") as f:
            json.dump(gs_report["rank_stats"], f, indent=2)
        with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump({key: value for key, value in vars(args).items() if key != "_wandb_run"}, f, indent=2)

    if distributed:
        model_engine = base.train_deepspeed(model, train_loader, args)
        model_to_save = model_engine.module
        eval_device = model_engine.device
    else:
        base.train(model, train_loader, args, device)
        model_to_save = model
        eval_device = device

    base.barrier()
    if base.is_main_process():
        model_to_save.save_pretrained(os.path.join(args.output_dir, "last_adapter"))
        tokenizer.save_pretrained(args.output_dir)
        metrics = evaluate_humaneval(model_to_save, tokenizer, eval_dataset, args, eval_device)
        predictions = metrics.pop("predictions")
        base.log_wandb(args, {f"humaneval/{key}": value for key, value in metrics.items()})
        with open(os.path.join(args.output_dir, "humaneval_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        with open(os.path.join(args.output_dir, "humaneval_predictions.jsonl"), "w", encoding="utf-8") as f:
            for item in predictions:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        base.finish_wandb(args)
    base.barrier()


if __name__ == "__main__":
    main()
