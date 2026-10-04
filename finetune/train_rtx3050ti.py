"""Fine-tune VieNeu-TTS-0.3B bằng LoRA trên RTX 3050 Ti Laptop 4 GB.

Yêu cầu đầu vào: finetune/dataset/metadata_encoded.csv
Chạy thử: uv run python finetune/train_rtx3050ti.py --smoke-test
Chạy chính: uv run python finetune/train_rtx3050ti.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset, Subset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from vieneu_utils.phonemize_text import phonemize_with_dict  # noqa: E402


class VieNeuEncodedDataset(Dataset):
    def __init__(self, metadata_path: Path, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: list[tuple[str, list[int]]] = []

        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy {metadata_path}. Hãy chạy filter_data.py và encode_data.py trước."
            )

        with metadata_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                parts = line.rstrip("\n").split("|", maxsplit=2)
                if len(parts) != 3:
                    print(f"Bỏ qua dòng {line_number}: sai định dạng filename|text|codes")
                    continue
                try:
                    codes = json.loads(parts[2])
                except json.JSONDecodeError:
                    print(f"Bỏ qua dòng {line_number}: codes không phải JSON hợp lệ")
                    continue
                if parts[1].strip() and isinstance(codes, list) and codes:
                    self.samples.append((parts[1].strip(), codes))

        if len(self.samples) < 2:
            raise ValueError("Cần ít nhất 2 mẫu đã encode để chia train/validation.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        text, codes = self.samples[index]
        try:
            phones = phonemize_with_dict(text)
        except Exception as exc:
            print(f"Cảnh báo: phonemize lỗi ({exc}); dùng văn bản gốc.")
            phones = text

        code_text = "".join(f"<|speech_{int(code)}|>" for code in codes)
        prompt = (
            f"<|TEXT_PROMPT_START|>{phones}<|TEXT_PROMPT_END|>"
            f"<|SPEECH_GENERATION_START|>{code_text}<|SPEECH_GENERATION_END|>"
        )
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = torch.full_like(input_ids, -100)

        speech_start_id = self.tokenizer.convert_tokens_to_ids(
            "<|SPEECH_GENERATION_START|>"
        )
        positions = (input_ids == speech_start_id).nonzero(as_tuple=True)[0]
        if len(positions):
            start = int(positions[0])
            labels[start:] = input_ids[start:]
            labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA VieNeu-TTS cho RTX 3050 Ti")
    parser.add_argument("--model", default="pnnbao-ump/VieNeu-TTS-0.3B")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "finetune" / "dataset" / "metadata_encoded.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "finetune" / "output" / "rtx3050ti-lora",
    )
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Chạy 20 step để kiểm tra toàn bộ pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Không thấy CUDA. Hãy cài nhóm dependency GPU trước khi train.")

    gpu_name = torch.cuda.get_device_name(0)
    vram_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name} ({vram_gib:.1f} GiB VRAM)")
    if vram_gib < 3.5:
        raise RuntimeError("GPU có dưới 3.5 GiB VRAM; cấu hình này không đủ an toàn.")

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = VieNeuEncodedDataset(args.dataset, tokenizer, args.max_length)
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    validation_size = max(1, round(len(indices) * args.val_ratio))
    validation_size = min(validation_size, len(indices) - 1)
    eval_dataset = Subset(dataset, indices[:validation_size])
    train_dataset = Subset(dataset, indices[validation_size:])
    print(f"Dữ liệu: {len(train_dataset)} train / {len(eval_dataset)} validation")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    max_steps = 20 if args.smoke_test else args.max_steps
    train_args = TrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        fp16=True,
        bf16=False,
        logging_steps=5 if args.smoke_test else 25,
        eval_strategy="steps",
        eval_steps=10 if args.smoke_test else 100,
        save_strategy="steps",
        save_steps=10 if args.smoke_test else 100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        optim="adamw_torch",
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir / "best_adapter"))
    tokenizer.save_pretrained(str(args.output_dir / "best_adapter"))
    print(f"Đã lưu LoRA adapter tại: {args.output_dir / 'best_adapter'}")


if __name__ == "__main__":
    main()
