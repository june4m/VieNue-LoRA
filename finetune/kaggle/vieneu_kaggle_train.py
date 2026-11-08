"""Fine-tune VieNeu-TTS-0.3B bằng LoRA trên Kaggle (GPU T4).

Kernel này chạy trọn pipeline trong một lần:
  1. Cài dependency (không đụng tới torch có sẵn của Kaggle).
  2. Lấy mã nguồn repo ở một commit cố định để dùng đúng phonemizer của API.
  3. Tải dataset HF, lọc 3–15 giây + rule text của repo, encode NeuCodec.
  4. Train LoRA (fp16 trên T4, validation split có seed, chọn checkpoint theo eval_loss).
  5. Sinh audio mẫu (câu hỏi/đáp án) bằng base model và adapter để nghe so sánh.
  6. Ghi adapter + voices.json + run_note.json vào /kaggle/working.

Đổi RUN_MODE = "smoke" -> "full" sau khi smoke test đạt.
Xem finetune/kaggle/README.md để biết lệnh push/theo dõi/tải kết quả.
"""

from __future__ import annotations

import os

# Phải đặt trước khi import torch: T4 x2 mà để Trainer thấy 2 GPU thì nó bọc
# DataParallel cùng gradient checkpointing — model 0.3B không cần card thứ hai.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import glob
import importlib.metadata as md
import io
import json
import math
import platform
import random
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

# ============================================================================
# CẤU HÌNH
# ============================================================================
RUN_MODE = "full"  # "smoke" (20 step kiểm tra pipeline) | "full" (train thật)

REPO_URL = "https://github.com/june4m/VieNue-LoRA.git"
REPO_REF = "c68d7d022a441669fdaf94e761fa6b212b764827"  # commit trên origin/main chứa phonemizer
BASE_MODEL = "pnnbao-ump/VieNeu-TTS-0.3B"
BASE_MODEL_REVISION = "4c4056e33bea543dd62115abc84c1612be8166c9"
CODEC_REPO = "neuphonic/neucodec"
HF_DATASET = "pnnbao-ump/ngochuyen_voice"
VOICE_NAME = "ngoc_huyen"
VOICE_DESCRIPTION = "Giọng nữ Ngọc Huyền (dataset pnnbao-ump/ngochuyen_voice, CC BY-NC 4.0)"

SEED = 42
MIN_SECONDS, MAX_SECONDS = 3.0, 15.0
MAX_TOKENS = 2048  # mẫu dài hơn bị LOẠI (không cắt) để model luôn thấy token kết thúc
VAL_RATIO = 0.05
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.05
MAX_TRAIN_HOURS = 9.5  # dừng + lưu trước giới hạn 12h của một phiên Kaggle

# Dùng lại metadata_encoded.csv từ output của lần chạy trước nếu được gắn làm input
# (kernel_sources trong kernel-metadata.json) -> bỏ qua bước tải + encode.
REUSE_ENCODED = True
# Tiếp tục từ checkpoint (vd "/kaggle/input/<kernel>/vieneu-lora/<run>/checkpoints/checkpoint-600").
RESUME_FROM: str | None = None

PROFILES = {
    "smoke": dict(
        data_files=["data/train-00000-of-00008.parquet"],  # 1/8 dataset cho nhanh
        max_samples=160,
        max_steps=20,
        epochs=None,
        batch_size=4,
        grad_accum=1,
        eval_steps=10,
        logging_steps=5,
        min_val=8,
    ),
    "full": dict(
        data_files=None,  # toàn bộ 8 shard
        max_samples=None,  # None = encode mọi clip qua bộ lọc
        max_steps=None,  # tính từ epochs
        epochs=3,
        batch_size=4,
        grad_accum=2,
        eval_steps=None,  # tự tính ~12 lần eval mỗi run
        logging_steps=25,
        min_val=50,
    ),
}

EVAL_PROMPTS = [
    "Câu hỏi số một. Thủ đô của nước Việt Nam là thành phố nào?",
    "Đáp án A: Hà Nội. Đáp án B: Huế. Đáp án C: Đà Nẵng. Đáp án D: Thành phố Hồ Chí Minh.",
    "Bạn đã chọn đáp án B. Bạn có muốn xác nhận câu trả lời này không?",
    "Câu 2: Tính giá trị của biểu thức 15 cộng 27, rồi chọn kết quả đúng.",
]

WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd() / "kaggle_working"
REPO_DIR = Path("/tmp/vieneu_repo")
START_TIME = time.time()


def log(msg: str = "") -> None:
    elapsed = time.time() - START_TIME
    print(f"[{elapsed / 60:6.1f}m] {msg}", flush=True)


def run(cmd: list[str]) -> None:
    log("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def pkg_version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


# ============================================================================
# 1. MÔI TRƯỜNG
# ============================================================================
def preflight() -> None:
    """Báo lỗi ngay nếu Kaggle không cấp GPU/Internet (thường do tài khoản chưa xác minh SĐT)."""
    import shutil
    import socket

    problems = []
    torch_ver = pkg_version("torch") or ""
    if torch_ver.endswith("+cpu") or not shutil.which("nvidia-smi"):
        problems.append(f"không có GPU (torch {torch_ver or 'chưa cài'}, nvidia-smi {'có' if shutil.which('nvidia-smi') else 'không có'})")
    try:
        socket.getaddrinfo("pypi.org", 443)
    except OSError as exc:
        problems.append(f"không có Internet ({exc})")
    if problems:
        raise RuntimeError(
            "Kernel " + " và ".join(problems) + ". Kaggle chỉ bật GPU + Internet cho tài khoản đã xác minh "
            "số điện thoại (kaggle.com/settings → Phone Verification); sau đó push lại với "
            "--accelerator NvidiaTeslaT4."
        )
    log("Preflight OK: có GPU và Internet")


def setup_hf_access() -> None:
    """neuphonic/neucodec là repo gated: cần secret HF_TOKEN của tài khoản đã đồng ý điều khoản."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient

            token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception as exc:
            log(f"Không đọc được Kaggle Secret HF_TOKEN: {type(exc).__name__}")
    if not token:
        raise RuntimeError(
            "Thiếu HF_TOKEN. Mở notebook trên Kaggle → Add-ons → Secrets → thêm secret tên HF_TOKEN "
            "(token Read của tài khoản HF đã bấm đồng ý ở huggingface.co/neuphonic/neucodec) và bật nó cho notebook."
        )
    os.environ["HF_TOKEN"] = token
    from huggingface_hub import HfApi

    try:
        HfApi().auth_check(CODEC_REPO, token=token)
    except Exception as exc:
        raise RuntimeError(
            f"HF_TOKEN chưa có quyền vào {CODEC_REPO} ({type(exc).__name__}). "
            f"Bấm đồng ý điều khoản tại https://huggingface.co/{CODEC_REPO} bằng đúng tài khoản tạo token."
        ) from None
    log(f"HF_TOKEN OK: có quyền truy cập {CODEC_REPO}")


def install_dependencies() -> dict:
    """Cài peft/neucodec/sea-g2p nhưng khóa cứng bộ torch sẵn có của Kaggle."""
    torch_family = {name: pkg_version(name) for name in ("torch", "torchaudio", "torchvision")}
    log(f"Torch có sẵn: {torch_family}")
    if not torch_family["torch"]:
        raise RuntimeError("Image Kaggle không có torch — hãy chọn môi trường GPU mặc định.")

    constraints = Path("/tmp/torch_constraints.txt")
    constraints.write_text(
        "".join(f"{name}=={ver}\n" for name, ver in torch_family.items() if ver),
        encoding="utf-8",
    )
    # neucodec -> torchtune 0.6.1 (đã ngừng phát triển) import torchao.dtypes.nf4tensor,
    # module này bị bỏ từ torchao 0.18 -> ghim 0.17.0 (bản đang chạy được ở venv local).
    packages = ["peft", "neucodec", "torchao==0.17.0", "sea-g2p", "librosa", "soundfile"]
    if not pkg_version("datasets"):
        packages.append("datasets")
    run([sys.executable, "-m", "pip", "install", "-q", *packages, "-c", str(constraints)])
    # Kiểm tra import ngay để lỗi dependency lộ ra trước khi tải dataset.
    run([sys.executable, "-c", "from neucodec import NeuCodec; import peft, sea_g2p"])

    after = {name: pkg_version(name) for name in torch_family}
    if after != torch_family:
        raise RuntimeError(f"pip đã thay đổi bộ torch: {torch_family} -> {after}")
    return torch_family


def fetch_repo() -> str:
    """Lấy repo ở commit cố định (để phonemizer khớp với engine dùng trong API)."""
    if not (REPO_DIR / ".git").exists():
        REPO_DIR.mkdir(parents=True, exist_ok=True)
        run(["git", "-C", str(REPO_DIR), "init", "-q"])
        run(["git", "-C", str(REPO_DIR), "remote", "add", "origin", REPO_URL])
    run(["git", "-C", str(REPO_DIR), "fetch", "-q", "--depth", "1", "origin", REPO_REF])
    run(["git", "-C", str(REPO_DIR), "checkout", "-q", "FETCH_HEAD"])
    sha = subprocess.run(
        ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    sys.path.insert(0, str(REPO_DIR / "src"))
    log(f"Repo {REPO_URL} @ {sha}")
    return sha


# ============================================================================
# 2. DỮ LIỆU
# ============================================================================
ACRONYM = re.compile(r"(?:[a-zA-Z]\.){2,}")
ACRONYM_NO_PERIOD = re.compile(r"(?:[A-Z]){2,}")


def text_ok(text: str) -> bool:
    """Cùng rule với finetune/data_scripts/filter_data.py (+ cấm ký tự phân cách)."""
    if not text or "|" in text or "\n" in text:
        return False
    if re.search(r"\d", text):
        return False
    if ACRONYM.search(text) or ACRONYM_NO_PERIOD.search(text):
        return False
    return text[-1] in ".,?!"


def find_reusable_encoded(profile: dict) -> Path | None:
    """Chỉ dùng lại khi CÙNG dataset, codec, shard và giới hạn mẫu — tránh để run full
    vô tình train trên 160 clip của smoke test."""
    if not REUSE_ENCODED:
        return None
    for path in sorted(glob.glob("/kaggle/input/**/metadata_encoded.csv", recursive=True)):
        info = Path(path).with_name("encode_info.json")
        if not info.exists():
            continue
        meta = json.loads(info.read_text(encoding="utf-8"))
        same = (
            meta.get("dataset") == HF_DATASET
            and meta.get("codec") == CODEC_REPO
            and meta.get("data_files") == profile["data_files"]
            and meta.get("max_samples") == profile["max_samples"]
        )
        if same:
            return Path(path)
        log(f"Bỏ qua {path}: encode_info không khớp profile {RUN_MODE}")
    return None


def build_encoded_metadata(profile: dict, torch, data_dir: Path) -> tuple[Path, dict]:
    import librosa
    import numpy as np
    import soundfile as sf
    from datasets import Audio, load_dataset

    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "metadata_encoded.csv"

    log(f"Tải dataset {HF_DATASET} (data_files={profile['data_files']})")
    kwargs = {"split": "train"}
    if profile["data_files"]:
        # Tập con: bỏ kiểm tra kích thước split (card ghi số liệu của cả 8 shard).
        kwargs["data_files"] = {"train": profile["data_files"]}
        kwargs["verification_mode"] = "no_checks"
    ds = load_dataset(HF_DATASET, **kwargs)
    ds = ds.cast_column("audio", Audio(decode=False))
    log(f"Dataset có {len(ds)} dòng")

    # Lượt 1: lọc theo text + thời lượng (chỉ đọc header audio)
    candidates, rejected = [], {"text": 0, "duration": 0, "audio_error": 0}
    texts = list(ds["transcription"])
    for idx in range(len(ds)):
        text = (texts[idx] or "").strip()
        if not text_ok(text):
            rejected["text"] += 1
            continue
        try:
            duration = sf.info(io.BytesIO(ds[idx]["audio"]["bytes"])).duration
        except Exception:
            rejected["audio_error"] += 1
            continue
        if not (MIN_SECONDS <= duration <= MAX_SECONDS):
            rejected["duration"] += 1
            continue
        candidates.append((idx, text, duration))
    log(f"Qua bộ lọc: {len(candidates)} | loại: {rejected}")

    random.Random(SEED).shuffle(candidates)
    if profile["max_samples"]:
        candidates = candidates[: profile["max_samples"]]
    log(f"Sẽ encode {len(candidates)} clip ({sum(c[2] for c in candidates) / 3600:.2f} giờ)")

    from neucodec import NeuCodec

    codec = NeuCodec.from_pretrained(CODEC_REPO).to("cuda").eval()
    rows, skipped, total_sec = [], 0, 0.0
    t0 = time.time()
    for n, (idx, text, duration) in enumerate(candidates, start=1):
        sample = ds[idx]
        filename = os.path.basename(sample.get("file_name") or f"sample_{idx:05d}.wav")
        try:
            wav, sr = sf.read(io.BytesIO(sample["audio"]["bytes"]), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != 16000:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            wav_tensor = torch.from_numpy(np.ascontiguousarray(wav)).float()[None, None, :]
            with torch.no_grad():
                codes = codec.encode_code(wav_tensor)
            codes = [int(x) for x in codes.squeeze().cpu().numpy().flatten().tolist()]
            if not codes or not all(0 <= c < 65536 for c in codes):
                skipped += 1
                continue
            rows.append(f"{filename}|{text}|{json.dumps(codes)}\n")
            total_sec += duration
        except Exception as exc:
            skipped += 1
            log(f"Lỗi encode {filename}: {exc}")
        if n % 250 == 0:
            rate = n / (time.time() - t0)
            log(f"Encode {n}/{len(candidates)} ({rate:.1f} clip/s)")

    out_path.write_text("".join(rows), encoding="utf-8")
    info = {
        "dataset": HF_DATASET,
        "data_files": profile["data_files"],
        "codec": CODEC_REPO,
        "max_samples": profile["max_samples"],
        "neucodec_version": pkg_version("neucodec"),
        "seed": SEED,
        "min_seconds": MIN_SECONDS,
        "max_seconds": MAX_SECONDS,
        "filter_rejected": rejected,
        "candidates_after_filter": len(candidates),
        "encoded_rows": len(rows),
        "encode_skipped": skipped,
        "encoded_hours": round(total_sec / 3600, 3),
        "encode_minutes": round((time.time() - t0) / 60, 1),
    }
    (data_dir / "encode_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Đã encode {len(rows)} dòng, bỏ qua {skipped} -> {out_path}")

    del codec
    torch.cuda.empty_cache()
    return out_path, info


def load_encoded(path: Path) -> list[dict]:
    samples = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.rstrip("\n").split("|", maxsplit=2)
            if len(parts) != 3:
                log(f"Bỏ qua dòng {line_number}: sai định dạng filename|text|codes")
                continue
            codes = json.loads(parts[2])
            if parts[1].strip() and isinstance(codes, list) and codes:
                samples.append({"filename": parts[0], "text": parts[1].strip(), "codes": codes})
    return samples


# ============================================================================
# 3. TOKEN HÓA
# ============================================================================
def build_features(samples, tokenizer, phonemize, special):
    """Giống train_rtx3050ti.py: chỉ tính loss từ <|SPEECH_GENERATION_START|> trở đi."""
    features, dropped_long, dropped_no_end, phoneme_errors = [], 0, 0, 0
    for sample in samples:
        try:
            phones = phonemize(sample["text"])
        except Exception:
            phoneme_errors += 1
            phones = sample["text"]
        code_text = "".join(f"<|speech_{int(c)}|>" for c in sample["codes"])
        prompt = (
            f"<|TEXT_PROMPT_START|>{phones}<|TEXT_PROMPT_END|>"
            f"<|SPEECH_GENERATION_START|>{code_text}<|SPEECH_GENERATION_END|>"
        )
        ids = tokenizer.encode(prompt)
        if len(ids) > MAX_TOKENS:
            dropped_long += 1
            continue
        if ids[-1] != special["end"] or special["start"] not in ids:
            dropped_no_end += 1
            continue
        start = ids.index(special["start"])
        labels = [-100] * start + ids[start:]
        features.append(
            {
                "input_ids": ids,
                "labels": labels,
                "phones": phones,
                "text": sample["text"],
                "codes": sample["codes"],
                "filename": sample["filename"],
            }
        )
    stats = {
        "kept": len(features),
        "dropped_over_max_tokens": dropped_long,
        "dropped_missing_special_tokens": dropped_no_end,
        "phonemize_errors": phoneme_errors,
    }
    if features:
        lengths = sorted(len(f["input_ids"]) for f in features)
        stats.update(
            min_tokens=lengths[0],
            mean_tokens=round(sum(lengths) / len(lengths), 1),
            p95_tokens=lengths[int(0.95 * (len(lengths) - 1))],
            max_tokens=lengths[-1],
        )
    return features, stats


class FeatureDataset:
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        f = self.features[idx]
        return {"input_ids": f["input_ids"], "labels": f["labels"]}


def make_collator(pad_id: int, torch):
    def collate(batch):
        width = max(len(item["input_ids"]) for item in batch)
        input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), width), -100, dtype=torch.long)
        attention = torch.zeros((len(batch), width), dtype=torch.long)
        for row, item in enumerate(batch):
            n = len(item["input_ids"])
            input_ids[row, :n] = torch.tensor(item["input_ids"], dtype=torch.long)
            labels[row, :n] = torch.tensor(item["labels"], dtype=torch.long)
            attention[row, :n] = 1
        return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}

    return collate


# ============================================================================
# 4. TRAIN
# ============================================================================
def filtered_training_args(TrainingArguments, **kwargs):
    """Bỏ các tham số mà bản transformers trên Kaggle không hỗ trợ (đổi tên qua các version)."""
    import inspect

    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" not in params and "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    dropped = [k for k in kwargs if k not in params]
    for key in dropped:
        kwargs.pop(key)
    if dropped:
        log(f"TrainingArguments bỏ qua tham số không hỗ trợ: {dropped}")
    return TrainingArguments(**kwargs)


def dtype_kwargs(torch_dtype):
    from packaging.version import Version

    if Version(pkg_version("transformers")) >= Version("4.56.0"):
        return {"dtype": torch_dtype}
    return {"torch_dtype": torch_dtype}


# ============================================================================
# 5. SINH AUDIO MẪU (bám theo vieneu/standard.py: _apply_chat_template + _infer_torch)
# ============================================================================
SPEECH_RE = re.compile(r"<\|speech_(\d+)\|>")


def synthesize(model, tokenizer, codec, phonemize, special, ref, text, torch, seed):
    ref_codes_str = "".join(f"<|speech_{int(c)}|>" for c in ref["codes"])
    phones = f"{ref['phones']} {phonemize(text)}"
    ids = (
        [special["text_start"]]
        + tokenizer.encode(phones, add_special_tokens=False)
        + [special["text_end"], special["start"]]
        + tokenizer.encode(ref_codes_str, add_special_tokens=False)
    )
    prompt = torch.tensor(ids, dtype=torch.long, device="cuda")[None, :]
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(
            input_ids=prompt,
            attention_mask=torch.ones_like(prompt),
            max_length=MAX_TOKENS,
            eos_token_id=special["end"],
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=1.0,
            top_k=50,
            min_new_tokens=50,
            use_cache=True,
        )
    new_tokens = out[0, prompt.shape[-1]:]
    hit_end = bool((new_tokens == special["end"]).any())
    speech_ids = [int(x) for x in SPEECH_RE.findall(tokenizer.decode(new_tokens.tolist()))]
    if not speech_ids:
        raise ValueError("Model không sinh ra speech token nào")
    with torch.no_grad():
        codec_device = next(codec.parameters()).device
        codes = torch.tensor(speech_ids, dtype=torch.long, device=codec_device)[None, None, :]
        wav = codec.decode_code(codes).float().cpu().numpy()[0, 0, :]
    return wav, {"speech_tokens": len(speech_ids), "hit_end_token": hit_end, "seconds": round(len(wav) / 24000, 2)}


def generate_samples(adapter_dir, tokenizer, phonemize, special, ref, torch, out_dir):
    import soundfile as sf
    from neucodec import NeuCodec
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    out_dir.mkdir(parents=True, exist_ok=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=BASE_MODEL_REVISION, **dtype_kwargs(torch.float32)
    ).to("cuda")
    model = PeftModel.from_pretrained(base, str(adapter_dir)).eval()
    codec = NeuCodec.from_pretrained(CODEC_REPO).to("cuda").eval()
    results = []
    for i, text in enumerate(EVAL_PROMPTS, start=1):
        for system in ("base", "lora"):
            name = f"{i:02d}_{system}.wav"
            try:
                if system == "base":
                    with model.disable_adapter():
                        wav, info = synthesize(model, tokenizer, codec, phonemize, special, ref, text, torch, SEED + i)
                else:
                    wav, info = synthesize(model, tokenizer, codec, phonemize, special, ref, text, torch, SEED + i)
                sf.write(str(out_dir / name), wav, 24000)
                results.append({"file": name, "text": text, "ok": True, **info})
                log(f"Mẫu {name}: {info}")
            except Exception as exc:
                results.append({"file": name, "text": text, "ok": False, "error": repr(exc)})
                log(f"Mẫu {name} LỖI: {exc!r}")
    (out_dir / "prompts.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    del model, base, codec
    torch.cuda.empty_cache()
    return results


# ============================================================================
# MAIN
# ============================================================================
def main() -> None:
    profile = PROFILES[RUN_MODE]
    run_name = f"{VOICE_NAME}-{RUN_MODE}-{time.strftime('%Y%m%d-%H%M')}-s{SEED}"
    run_dir = WORK / "vieneu-lora" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    note: dict = {"run_name": run_name, "run_mode": RUN_MODE, "status": "started"}
    note_path = run_dir / "run_note.json"

    def save_note():
        note_path.write_text(json.dumps(note, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    log(f"RUN_MODE={RUN_MODE} profile={profile} run={run_name}")
    try:
        preflight()
        setup_hf_access()
        note["torch_family"] = install_dependencies()
        note["repo_sha"] = fetch_repo()

        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainerCallback,
            TrainingArguments,
            set_seed,
        )
        from vieneu_utils.phonemize_text import phonemize_with_dict

        if not torch.cuda.is_available():
            raise RuntimeError("Kernel không có GPU. Bật Accelerator = GPU T4 x2 cho notebook.")
        capability = torch.cuda.get_device_capability(0)
        use_bf16 = capability >= (8, 0) and torch.cuda.is_bf16_supported()
        note["environment"] = {
            "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0),
            "gpu_count_visible": torch.cuda.device_count(),
            "vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1),
            "compute_capability": list(capability),
            "precision": "bf16" if use_bf16 else "fp16",
            **{p: pkg_version(p) for p in ("torch", "transformers", "peft", "neucodec", "datasets", "sea-g2p", "accelerate")},
        }
        log(f"Môi trường: {note['environment']}")
        save_note()
        set_seed(SEED)

        # ---------------- dữ liệu ----------------
        reuse = find_reusable_encoded(profile)
        if reuse:
            log(f"Dùng lại dữ liệu đã encode: {reuse}")
            encoded_path = reuse
            note["encode_info"] = json.loads(reuse.with_name("encode_info.json").read_text(encoding="utf-8"))
            note["encode_info"]["reused_from"] = str(reuse)
        else:
            encoded_path, note["encode_info"] = build_encoded_metadata(profile, torch, WORK / "data")
        save_note()

        samples = load_encoded(encoded_path)
        if profile["max_samples"]:
            samples = samples[: profile["max_samples"]]

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_MODEL_REVISION)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        special = {
            "text_start": tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_START|>"),
            "text_end": tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_END|>"),
            "start": tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>"),
            "end": tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>"),
        }
        note["tokens"] = {
            "pad": [tokenizer.pad_token, tokenizer.pad_token_id],
            "eos": [tokenizer.eos_token, tokenizer.eos_token_id],
            **special,
        }
        log(f"Token đặc biệt: {note['tokens']}")
        if special["end"] == tokenizer.pad_token_id:
            raise RuntimeError("pad_token trùng <|SPEECH_GENERATION_END|> — cần xử lý mask riêng.")

        features, note["token_stats"] = build_features(samples, tokenizer, phonemize_with_dict, special)
        log(f"Thống kê token: {note['token_stats']}")
        if len(features) < 20:
            raise RuntimeError(f"Chỉ còn {len(features)} mẫu sau token hóa — quá ít để train.")

        order = list(range(len(features)))
        random.Random(SEED).shuffle(order)
        n_val = min(max(profile["min_val"], round(len(order) * VAL_RATIO)), len(order) // 4)
        val_feats = [features[i] for i in order[:n_val]]
        train_feats = [features[i] for i in order[n_val:]]
        (run_dir / "split.json").write_text(
            json.dumps(
                {"seed": SEED, "train": [f["filename"] for f in train_feats], "validation": [f["filename"] for f in val_feats]},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        log(f"Split: {len(train_feats)} train / {len(val_feats)} validation")

        # ---------------- model + LoRA ----------------
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, revision=BASE_MODEL_REVISION, **dtype_kwargs(torch.float32)
        )
        model.config.use_cache = False
        model.enable_input_require_grads()
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        log(f"Tham số train: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

        effective_batch = profile["batch_size"] * profile["grad_accum"]
        if profile["max_steps"]:
            max_steps = profile["max_steps"]
        else:
            max_steps = math.ceil(profile["epochs"] * len(train_feats) / effective_batch)
        eval_steps = profile["eval_steps"] or max(50, max_steps // 12)
        note["hyperparameters"] = {
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "target_modules": sorted(lora_config.target_modules),
            "learning_rate": LEARNING_RATE,
            "warmup_steps": max(1, int(WARMUP_RATIO * max_steps)),
            "per_device_batch_size": profile["batch_size"],
            "gradient_accumulation_steps": profile["grad_accum"],
            "effective_batch_size": effective_batch,
            "max_steps": max_steps,
            "approx_epochs": round(max_steps * effective_batch / len(train_feats), 2),
            "eval_steps": eval_steps,
            "max_tokens": MAX_TOKENS,
            "seed": SEED,
            "trainable_params": trainable,
        }
        log(f"Siêu tham số: {note['hyperparameters']}")
        save_note()

        class TimeLimit(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if (time.time() - START_TIME) / 3600 > MAX_TRAIN_HOURS:
                    log(f"Chạm giới hạn {MAX_TRAIN_HOURS}h — lưu checkpoint và dừng")
                    control.should_save = True
                    control.should_training_stop = True
                return control

            def on_log(self, args, state, control, logs=None, **kwargs):
                peak = torch.cuda.max_memory_allocated() / 1024**3
                log(f"step {state.global_step}/{state.max_steps} {logs} | peak VRAM {peak:.2f} GiB")

        args = filtered_training_args(
            TrainingArguments,
            output_dir=str(run_dir / "checkpoints"),
            max_steps=max_steps,
            per_device_train_batch_size=profile["batch_size"],
            per_device_eval_batch_size=profile["batch_size"],
            gradient_accumulation_steps=profile["grad_accum"],
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=LEARNING_RATE,
            warmup_steps=note["hyperparameters"]["warmup_steps"],
            lr_scheduler_type="cosine",
            weight_decay=0.0,
            max_grad_norm=1.0,
            fp16=not use_bf16,
            bf16=use_bf16,
            logging_steps=profile["logging_steps"],
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=eval_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            group_by_length=True,
            report_to="none",
            disable_tqdm=True,
            dataloader_num_workers=2,
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            optim="adamw_torch",
            seed=SEED,
            data_seed=SEED,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=FeatureDataset(train_feats),
            eval_dataset=FeatureDataset(val_feats),
            data_collator=make_collator(tokenizer.pad_token_id, torch),
            callbacks=[TimeLimit()],
        )
        torch.cuda.reset_peak_memory_stats()
        train_start = time.time()
        result = trainer.train(resume_from_checkpoint=RESUME_FROM)
        final_eval = trainer.evaluate()

        adapter_dir = run_dir / "best_adapter"
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        note["training"] = {
            "train_minutes": round((time.time() - train_start) / 60, 1),
            "global_step": trainer.state.global_step,
            "train_loss": round(result.training_loss, 4),
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_eval_loss": trainer.state.best_metric,
            "final_eval": final_eval,
            "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
            "stopped_by_time_limit": trainer.state.global_step < max_steps,
        }
        (run_dir / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")
        log(f"Train xong: {note['training']}")
        save_note()
        del trainer, model
        torch.cuda.empty_cache()

        # ---------------- voices.json + audio mẫu ----------------
        val_by_duration = sorted(val_feats, key=lambda f: abs(len(f["codes"]) / 50 - 6.0))
        ref = val_by_duration[0]
        voices = {
            "meta": {
                "spec": "vieneu.voice.presets",
                "spec_version": "1.0",
                "engine": "VieNeu-TTS",
                "license": "CC BY-NC 4.0",
                "notice": "Base model pnnbao-ump/VieNeu-TTS-0.3B; dataset pnnbao-ump/ngochuyen_voice (non-commercial).",
            },
            "default_voice": VOICE_NAME,
            "presets": {VOICE_NAME: {"codes": ref["codes"], "text": ref["text"], "description": VOICE_DESCRIPTION}},
        }
        (adapter_dir / "voices.json").write_text(json.dumps(voices, ensure_ascii=False), encoding="utf-8")
        note["reference_voice"] = {"filename": ref["filename"], "text": ref["text"], "seconds": round(len(ref["codes"]) / 50, 2)}
        log(f"voices.json: ref {note['reference_voice']}")

        try:
            note["samples"] = generate_samples(
                adapter_dir, tokenizer, phonemize_with_dict, special, ref, torch, run_dir / "samples"
            )
            inference_ok = all(s["ok"] for s in note["samples"] if s["file"].endswith("_lora.wav"))
        except Exception:
            note["samples_error"] = traceback.format_exc()
            log(note["samples_error"])
            inference_ok = False

        loss_ok = math.isfinite(note["training"]["train_loss"]) and math.isfinite(final_eval.get("eval_loss", float("nan")))
        note["status"] = "passed" if (loss_ok and inference_ok) else "failed_checks"
        note["total_minutes"] = round((time.time() - START_TIME) / 60, 1)
        save_note()
        log(f"KẾT QUẢ {RUN_MODE.upper()}: {note['status']} | loss hữu hạn={loss_ok} | sinh audio={inference_ok}")
        log(f"Adapter: {adapter_dir}")
    except Exception:
        note["status"] = "error"
        note["error"] = traceback.format_exc()
        save_note()
        log(note["error"])
        raise


if __name__ == "__main__":
    main()
