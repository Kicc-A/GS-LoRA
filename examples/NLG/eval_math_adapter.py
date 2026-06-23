import argparse
import json
import os
import shutil
import tempfile
from types import SimpleNamespace

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_math_gslora import (
    apply_dataset_preset,
    apply_prompt_config,
    extract_gsm8k_answer,
    format_prompt,
    load_split,
    maybe_select,
    normalize_dataset_lengths,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved GS-LoRA/PEFT adapter on GSM8K.")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--backend", choices=["hf", "vllm"], default="vllm")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--eval_seed", type=int, default=0)
    parser.add_argument("--vllm_dtype", type=str, default="bfloat16")
    parser.add_argument("--vllm_max_lora_rank", type=int, default=32)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--run_config", type=str, default=None)
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--eval_dataset_name", type=str, default=None)
    parser.add_argument("--eval_dataset_config", type=str, default=None)
    parser.add_argument("--eval_split", type=str, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def infer_run_config(adapter_path: str):
    parent = os.path.dirname(os.path.abspath(adapter_path))
    if os.path.basename(os.path.abspath(adapter_path)) == "last_adapter":
        candidate = os.path.join(parent, "run_config.json")
    else:
        candidate = os.path.join(os.path.dirname(parent), "run_config.json")
    return candidate if os.path.exists(candidate) else None


def load_eval_args(cli_args):
    run_config = cli_args.run_config or infer_run_config(cli_args.adapter_path)
    config = {}
    if run_config:
        with open(run_config, "r", encoding="utf-8") as f:
            config.update(json.load(f))

    overrides = {
        "model_name_or_path": cli_args.model_name_or_path,
        "output_dir": cli_args.output_dir,
        "eval_dataset_name": cli_args.eval_dataset_name,
        "eval_dataset_config": cli_args.eval_dataset_config,
        "eval_split": cli_args.eval_split,
        "max_eval_samples": cli_args.max_eval_samples,
        "per_device_eval_batch_size": cli_args.per_device_eval_batch_size,
        "max_new_tokens": cli_args.max_new_tokens,
        "seed": cli_args.seed,
    }
    config.update({key: value for key, value in overrides.items() if value is not None})

    config.setdefault("output_dir", os.path.dirname(os.path.abspath(cli_args.adapter_path)))
    config.setdefault("eval_dataset_name", "gsm8k")
    config.setdefault("eval_dataset_config", "main")
    config.setdefault("eval_split", "test")
    config.setdefault("eval_question_column", "question")
    config.setdefault("eval_answer_column", "answer")
    config.setdefault("dataset_preset", "loraga_metamathqa")
    config.setdefault("dataset_meta_prompt", "")
    config.setdefault("dataset_prefix", "Q: ")
    config.setdefault("dataset_postfix", "\nA: ")
    config.setdefault("prompt_path", None)
    config.setdefault("max_length", 512)
    config.setdefault("max_src_len", 512)
    config.setdefault("loraga_filter_gsm", True)
    config.setdefault("loraga_max_tokens", 512)
    config.setdefault("max_eval_samples", None)
    config.setdefault("per_device_eval_batch_size", 1)
    config.setdefault("max_new_tokens", 256)
    config.setdefault("seed", 42)
    config.setdefault("trust_remote_code", False)

    if cli_args.trust_remote_code:
        config["trust_remote_code"] = True
    if cli_args.fp16:
        config["fp16"] = True
        config["bf16"] = False
    elif cli_args.bf16:
        config["bf16"] = True

    if not config.get("model_name_or_path"):
        adapter_config_path = os.path.join(cli_args.adapter_path, "adapter_config.json")
        with open(adapter_config_path, "r", encoding="utf-8") as f:
            adapter_config = json.load(f)
        config["model_name_or_path"] = adapter_config["base_model_name_or_path"]

    args = SimpleNamespace(**config)
    apply_prompt_config(args)
    apply_dataset_preset(args)
    normalize_dataset_lengths(args)
    return args



@torch.no_grad()
def evaluate_gsm8k_hf_sampling(model, tokenizer, eval_dataset, args, device):
    model.eval()
    correct = 0
    total = 0
    predictions = []

    from tqdm.auto import tqdm

    for start in tqdm(range(0, len(eval_dataset), args.per_device_eval_batch_size), desc="GSM8K eval"):
        batch = eval_dataset[start : start + args.per_device_eval_batch_size]
        questions = batch[args.eval_question_column]
        answers = batch[args.eval_answer_column]
        prompts = [format_prompt(question, args) for question in questions]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        prompt_length = inputs["input_ids"].shape[1]

        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        for index, output_ids in enumerate(generated):
            prediction_text = tokenizer.decode(output_ids[prompt_length:], skip_special_tokens=True)
            pred_answer = extract_gsm8k_answer(prediction_text)
            gold_answer = extract_gsm8k_answer(str(answers[index]))
            is_correct = pred_answer == gold_answer
            correct += int(is_correct)
            total += 1
            predictions.append(
                {
                    "question": questions[index],
                    "prediction": prediction_text,
                    "pred_answer": pred_answer,
                    "gold_answer": gold_answer,
                    "correct": is_correct,
                }
            )

    return {
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "total": total,
        "predictions": predictions,
    }


def patched_adapter_for_vllm(adapter_path):
    temp_dir = tempfile.mkdtemp(prefix="gslora_vllm_adapter_")
    for name in os.listdir(adapter_path):
        src = os.path.join(adapter_path, name)
        dst = os.path.join(temp_dir, name)
        if name == "adapter_config.json":
            with open(src, "r", encoding="utf-8") as f:
                adapter_config = json.load(f)
            adapter_config["task_type"] = "CAUSAL_LM"
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(adapter_config, f, indent=2)
            continue
        try:
            os.symlink(src, dst)
        except OSError:
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
    return temp_dir


def evaluate_gsm8k_vllm(eval_dataset, args, adapter_path):
    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except ImportError as exc:
        raise RuntimeError("vLLM is not installed in this Python environment. Use --backend hf, or run in the same env as LoRA-GA eval.") from exc

    model = LLM(
        args.model_name_or_path,
        dtype=args.vllm_dtype,
        seed=args.eval_seed,
        enable_lora=True,
        max_lora_rank=args.vllm_max_lora_rank,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
    )
    sampling_params = SamplingParams(
        top_p=args.top_p,
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
    )
    lora_request = LoRARequest("adapter", 1, patched_adapter_for_vllm(adapter_path))

    correct = 0
    total = 0
    predictions = []

    from tqdm.auto import tqdm

    for start in tqdm(range(0, len(eval_dataset), args.per_device_eval_batch_size), desc="GSM8K eval"):
        batch = eval_dataset[start : start + args.per_device_eval_batch_size]
        questions = batch[args.eval_question_column]
        answers = batch[args.eval_answer_column]
        prompts = [format_prompt(question, args) for question in questions]
        outputs = model.generate(prompts, sampling_params=sampling_params, use_tqdm=False, lora_request=lora_request)

        for question, answer, output in zip(questions, answers, outputs):
            prediction_text = output.outputs[0].text
            pred_answer = extract_gsm8k_answer(prediction_text)
            gold_answer = extract_gsm8k_answer(str(answer))
            is_correct = pred_answer == gold_answer
            correct += int(is_correct)
            total += 1
            predictions.append(
                {
                    "question": question,
                    "prediction": prediction_text,
                    "pred_answer": pred_answer,
                    "gold_answer": gold_answer,
                    "correct": is_correct,
                }
            )

    return {
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "total": total,
        "predictions": predictions,
    }

def main():
    cli_args = parse_args()
    args = load_eval_args(cli_args)
    set_seed(args.seed)

    args.temperature = cli_args.temperature
    args.top_p = cli_args.top_p
    args.eval_seed = cli_args.eval_seed
    args.vllm_dtype = cli_args.vllm_dtype
    args.vllm_max_lora_rank = cli_args.vllm_max_lora_rank
    args.vllm_gpu_memory_utilization = cli_args.vllm_gpu_memory_utilization
    set_seed(args.eval_seed)

    eval_dataset = load_split(args.eval_dataset_name, args.eval_dataset_config, args.eval_split)
    eval_dataset = maybe_select(eval_dataset, args.max_eval_samples, args.eval_seed)

    if cli_args.backend == "vllm":
        eval_metrics = evaluate_gsm8k_vllm(eval_dataset, args, cli_args.adapter_path)
    else:
        dtype = torch.bfloat16 if getattr(args, "bf16", False) else torch.float16 if getattr(args, "fp16", False) else torch.float32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=True,
            trust_remote_code=args.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=args.trust_remote_code,
        )
        peft_config = PeftConfig.from_pretrained(cli_args.adapter_path)
        peft_config.task_type = "CAUSAL_LM"
        model = PeftModel.from_pretrained(
            base_model,
            cli_args.adapter_path,
            is_trainable=False,
            config=peft_config,
        ).to(device)

        eval_metrics = evaluate_gsm8k_hf_sampling(model, tokenizer, eval_dataset, args, device)

    predictions = eval_metrics.pop("predictions")

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "gsm8k_metrics.json")
    predictions_path = os.path.join(args.output_dir, "gsm8k_predictions.jsonl")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)
    with open(predictions_path, "w", encoding="utf-8") as f:
        for item in predictions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(json.dumps(eval_metrics, indent=2))
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {predictions_path}")


if __name__ == "__main__":
    main()
