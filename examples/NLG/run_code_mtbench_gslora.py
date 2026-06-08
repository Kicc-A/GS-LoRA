import argparse
import json
import os
import time
import uuid
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

import run_math_gslora as base
from gs_lora import GSLoraConfig, prepare_gslora_model
from run_code_humaneval_gslora import tokenize_sft_example

LLAMA3_USER = "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
LLAMA3_ASSISTANT_PREFIX = "<|start_header_id|>assistant<|end_header_id|>\n\n"
LLAMA3_ASSISTANT = "<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a CausalLM with GS-LoRA on CodeFeedback and generate MTBench answers.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_dataset_name", type=str, default="m-a-p/CodeFeedback-Filtered-Instruction")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--train_question_column", type=str, default="query")
    parser.add_argument("--train_answer_column", type=str, default="answer")
    parser.add_argument("--mtbench_question_file", type=str, required=True)
    parser.add_argument("--mtbench_model_id", type=str, default="gslora-llama3.1-codefeedback")
    parser.add_argument("--target_modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    parser.add_argument("--target_prefix", type=str, default=None)
    parser.add_argument("--base_rank", type=int, default=8)
    parser.add_argument("--tau", type=float, default=0.90)
    parser.add_argument("--r_min", type=int, default=4)
    parser.add_argument("--r_max", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--use_lora_plus", action="store_true")
    parser.add_argument("--lora_plus_scaler", type=float, default=16.0)
    parser.add_argument("--init_method", choices=["none", "svd_sqrt", "svd_sigma", "svd_a_zero_b"], default="svd_a_zero_b")
    parser.add_argument("--init_scale", type=float, default=1.0)
    parser.add_argument("--no_compensate_scaling", action="store_true")
    parser.add_argument("--scaling_mode", choices=["rank", "sqrt_rank", "avg_rank"], default="rank")
    parser.add_argument("--calibration_steps", type=int, default=32)
    parser.add_argument("--skip_adaptive_rank", action="store_true")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
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
    parser.add_argument("--wandb_project", type=str, default="gslora-mtbench")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb_dir", type=str, default=None)
    parser.add_argument("--wandb_log_interval", type=int, default=10)
    return parser.parse_args()


def read_mtbench_questions(path: str) -> List[Dict[str, object]]:
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def build_prompt(history: List[Dict[str, str]]) -> str:
    chunks = []
    for message in history:
        if message["role"] == "user":
            chunks.append(LLAMA3_USER.format(content=message["content"].strip()))
        else:
            chunks.append(LLAMA3_ASSISTANT.format(content=message["content"].strip()))
    chunks.append(LLAMA3_ASSISTANT_PREFIX)
    return "".join(chunks)


def clean_generation(text: str) -> str:
    for stop in ("<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>user<|end_header_id|>"):
        if stop in text:
            text = text.split(stop)[0]
    return text.strip()


@torch.no_grad()
def generate_mtbench_answers(model, tokenizer, questions, args, device: torch.device) -> List[Dict[str, object]]:
    model.eval()
    answers = []
    for question in questions:
        history = []
        turns = []
        for user_turn in question["turns"]:
            history.append({"role": "user", "content": user_turn})
            prompt = build_prompt(history)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
            prompt_len = int(inputs["attention_mask"].sum(dim=1)[0].item())
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )[0]
            raw = tokenizer.decode(output_ids[prompt_len:], skip_special_tokens=False)
            answer = clean_generation(raw)
            turns.append(answer)
            history.append({"role": "assistant", "content": answer})
        answers.append(
            {
                "question_id": question["question_id"],
                "answer_id": uuid.uuid4().hex,
                "model_id": args.mtbench_model_id,
                "choices": [{"index": 0, "turns": turns}],
                "tstamp": time.time(),
            }
        )
    return answers


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
    train_dataset = base.maybe_select(train_dataset, args.max_train_samples, args.seed)
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
        compensate_scaling=not args.no_compensate_scaling,
        scaling_mode=args.scaling_mode,
    )

    def loss_fn(current_model, batch):
        batch = {key: value.to(device) for key, value in batch.items()}
        return current_model(**batch).loss

    model, gs_report = prepare_gslora_model(model=model, dataloader=calibration_loader, loss_fn=loss_fn, config=config, device=device)
    if base.is_main_process():
        if hasattr(model, "print_trainable_parameters"):
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
        questions = read_mtbench_questions(args.mtbench_question_file)
        answers = generate_mtbench_answers(model_to_save, tokenizer, questions, args, eval_device)
        answer_dir = os.path.join(args.output_dir, "mt_bench", "model_answer")
        os.makedirs(answer_dir, exist_ok=True)
        answer_file = os.path.join(answer_dir, f"{args.mtbench_model_id}.jsonl")
        with open(answer_file, "w", encoding="utf-8") as f:
            for answer in answers:
                f.write(json.dumps(answer, ensure_ascii=False) + "\n")
        metrics = {"num_questions": len(questions), "num_answers": len(answers)}
        base.log_wandb(args, {f"mtbench/{key}": value for key, value in metrics.items()})
        with open(os.path.join(args.output_dir, "mtbench_generation_metrics.json"), "w", encoding="utf-8") as f:
            json.dump({**metrics, "answer_file": answer_file}, f, indent=2)
        base.finish_wandb(args)
    base.barrier()


if __name__ == "__main__":
    main()
