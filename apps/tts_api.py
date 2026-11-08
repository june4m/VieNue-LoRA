"""
VieNeu-TTS API cho LearnAble — sinh audio câu hỏi / đáp án thi.
================================================================
Nạp backbone PyTorch ``pnnbao-ump/VieNeu-TTS-0.3B`` + LoRA adapter (train bằng
``finetune/kaggle``) MỘT lần khi khởi động, rồi phục vụ:

    GET  /health      trạng thái model (503 khi đang nạp / lỗi nạp)
    GET  /v1/voices   giọng có sẵn (lấy từ voices.json của adapter)
    POST /v1/tts      {"text": "...", "voice": "ngoc_huyen"?} -> audio/wav (PCM16, 24 kHz)

Audio được cache trên đĩa theo hash(model + giọng + text + tham số sinh), nên cùng
một câu hỏi luôn trả về đúng một file (sampling ngẫu nhiên không làm audio đổi
giữa các lần gọi) và lần gọi lại không tốn GPU.

Chạy (MỘT worker — mỗi worker là một bản model riêng trong RAM/VRAM):

    VIENEU_LORA=finetune/output/kaggle/vieneu-lora/<run>/best_adapter \\
    HF_TOKEN=hf_... python -m apps.tts_api          # http://0.0.0.0:8000

Biến môi trường: xem ``Settings.from_env``.
"""
# KHÔNG dùng `from __future__ import annotations`: annotation dạng chuỗi của TTSRequest
# chỉ resolve được khi module nằm đúng trong sys.modules. Chạy qua runpy.run_module(...,
# run_name="__main__") (launcher import pyarrow trước trên Windows) thì mọi POST /v1/tts trả 500.
import hashlib
import hmac
import io
import json
import logging
import os
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import soundfile as sf
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

logger = logging.getLogger("Vieneu.TTSApi")

SAMPLE_RATE = 24_000


# ---------------------------------------------------------------------------
# Cấu hình
# ---------------------------------------------------------------------------
def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    backbone: str = "pnnbao-ump/VieNeu-TTS-0.3B"
    lora: Optional[str] = None  # thư mục adapter hoặc HF repo id; None = base model
    codec: str = "neuphonic/neucodec"
    device: str = "auto"  # auto | cuda | cpu
    codec_device: Optional[str] = None  # None = giống device
    hf_token: Optional[str] = None
    api_key: Optional[str] = None  # None = không yêu cầu X-API-Key
    cache_dir: Optional[Path] = Path("tts_cache")  # None = tắt cache
    max_chars: int = 2000
    temperature: float = 1.0
    top_k: int = 50
    watermark: bool = True
    # LoRA được train KHÔNG có tag cảm xúc ở đầu prompt; engine standard mặc định
    # (emotion="natural") lại chèn <|emotion_0|>. Để False cho khớp lúc train.
    emotion_tag: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    @classmethod
    def from_env(cls) -> "Settings":
        cache = os.environ.get("VIENEU_CACHE_DIR", "tts_cache").strip()
        return cls(
            backbone=os.environ.get("VIENEU_BACKBONE", cls.backbone),
            lora=os.environ.get("VIENEU_LORA") or None,
            codec=os.environ.get("VIENEU_CODEC", cls.codec),
            device=os.environ.get("VIENEU_DEVICE", cls.device),
            codec_device=os.environ.get("VIENEU_CODEC_DEVICE") or None,
            hf_token=os.environ.get("HF_TOKEN") or None,
            api_key=os.environ.get("VIENEU_API_KEY") or None,
            cache_dir=Path(cache) if cache and cache.lower() != "off" else None,
            max_chars=int(os.environ.get("VIENEU_MAX_CHARS", cls.max_chars)),
            temperature=float(os.environ.get("VIENEU_TEMPERATURE", cls.temperature)),
            top_k=int(os.environ.get("VIENEU_TOP_K", cls.top_k)),
            watermark=_env_bool("VIENEU_WATERMARK", cls.watermark),
            emotion_tag=_env_bool("VIENEU_EMOTION_TAG", cls.emotion_tag),
            host=os.environ.get("VIENEU_HOST", cls.host),
            port=int(os.environ.get("VIENEU_PORT", cls.port)),
        )


# ---------------------------------------------------------------------------
# Nạp engine thật (chỉ import vieneu/torch ở đây để test không cần torch)
# ---------------------------------------------------------------------------
def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _check_lora_config(engine: Any, lora: str) -> None:
    """Adapter train bằng peft mới hơn bản trên server: đảm bảo r/alpha được đọc đúng."""
    config_path = Path(lora) / "adapter_config.json"
    if not config_path.is_file():
        return
    expected = json.loads(config_path.read_text(encoding="utf-8"))
    loaded = engine.backbone.peft_config["default"]
    if (loaded.r, loaded.lora_alpha) != (expected.get("r"), expected.get("lora_alpha")):
        raise RuntimeError(
            f"LoRA config đọc sai: r={loaded.r}, alpha={loaded.lora_alpha} "
            f"(adapter_config.json: r={expected.get('r')}, alpha={expected.get('lora_alpha')}). "
            "Hãy nâng cấp peft trên server."
        )


def load_engine(settings: Settings) -> Any:
    from vieneu import Vieneu

    device = _resolve_device(settings.device)
    codec_device = settings.codec_device or device
    logger.info(
        "Nạp backbone %s (%s), codec %s (%s), LoRA %s",
        settings.backbone, device, settings.codec, codec_device, settings.lora or "-",
    )
    engine = Vieneu(
        mode="standard",
        backbone_repo=settings.backbone,
        backbone_device=device,
        codec_repo=settings.codec,
        codec_device=codec_device,
        hf_token=settings.hf_token,
        gguf_filename=None,  # mặc định của engine là file GGUF -> phải tắt để dùng PyTorch
        emotion="natural" if settings.emotion_tag else "none",
    )
    if settings.lora:
        engine.load_lora_adapter(settings.lora, hf_token=settings.hf_token)
        _check_lora_config(engine, settings.lora)
    if not settings.watermark:
        engine.watermarker = None
    logger.info(
        "Engine sẵn sàng: default_voice=%s, emotion_tag=%s, voices=%s",
        getattr(engine, "_default_voice", None),
        getattr(engine, "default_emotion", None),
        [vid for _, vid in engine.list_preset_voices()],
    )
    return engine


# ---------------------------------------------------------------------------
# Dịch vụ: khóa, cache, mã hóa WAV
# ---------------------------------------------------------------------------
def _fingerprint_lora(lora: Optional[str]) -> str:
    """Định danh adapter cho cache key: hash file weights nếu là thư mục local."""
    if not lora:
        return "base"
    weights = Path(lora) / "adapter_model.safetensors"
    if weights.is_file():
        digest = hashlib.sha256()
        with weights.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()[:16]
    return lora  # HF repo id


def encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    pcm = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
    buffer = io.BytesIO()
    sf.write(buffer, pcm, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


@dataclass
class SynthesisResult:
    wav: bytes
    cache_key: str
    cache_hit: bool
    duration_s: float
    elapsed_ms: int
    voice: str


@dataclass
class TTSService:
    settings: Settings
    engine: Any = None
    load_error: Optional[str] = None
    model_version: str = "unloaded"
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def attach(self, engine: Any) -> None:
        self.engine = engine
        self.model_version = f"{self.settings.backbone}@{_fingerprint_lora(self.settings.lora)}"

    @property
    def ready(self) -> bool:
        return self.engine is not None

    @property
    def sample_rate(self) -> int:
        return int(getattr(self.engine, "sample_rate", SAMPLE_RATE))

    def default_voice(self) -> Optional[str]:
        voice = getattr(self.engine, "_default_voice", None)
        if voice:
            return voice
        voices = self.voices()
        return voices[0]["id"] if voices else None

    def voices(self) -> list[dict[str, str]]:
        return [{"id": vid, "description": desc} for desc, vid in self.engine.list_preset_voices()]

    def cache_key(self, text: str, voice: str) -> str:
        payload = json.dumps(
            {
                "model": self.model_version,
                "voice": voice,
                "text": text,
                "temperature": self.settings.temperature,
                "top_k": self.settings.top_k,
                "emotion_tag": self.settings.emotion_tag,
                "watermark": self.settings.watermark,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Optional[Path]:
        if self.settings.cache_dir is None:
            return None
        return self.settings.cache_dir / key[:2] / f"{key}.wav"

    def _write_cache(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def synthesize(self, text: str, voice: Optional[str]) -> SynthesisResult:
        start = time.perf_counter()
        voice_id = voice or self.default_voice()
        if voice_id is None:
            raise LookupError("Model không có giọng nào (thiếu voices.json trong adapter)")
        try:
            voice_data = self.engine.get_preset_voice(voice_id)
        except ValueError as exc:
            raise LookupError(str(exc)) from exc

        key = self.cache_key(text, voice_id)
        path = self._cache_path(key)
        if path is not None and path.is_file():
            data = path.read_bytes()
            return SynthesisResult(
                wav=data,
                cache_key=key,
                cache_hit=True,
                duration_s=sf.info(io.BytesIO(data)).duration,
                elapsed_ms=int((time.perf_counter() - start) * 1000),
                voice=voice_id,
            )

        # Model PyTorch không an toàn đa luồng -> sinh tuần tự.
        with self._lock:
            audio = self.engine.infer(
                text,
                voice=voice_data,
                temperature=self.settings.temperature,
                top_k=self.settings.top_k,
                apply_watermark=self.settings.watermark,
            )
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            raise RuntimeError("Engine trả về audio rỗng")
        data = encode_wav(audio, self.sample_rate)
        if path is not None:
            self._write_cache(path, data)
        return SynthesisResult(
            wav=data,
            cache_key=key,
            cache_hit=False,
            duration_s=audio.size / self.sample_rate,
            elapsed_ms=int((time.perf_counter() - start) * 1000),
            voice=voice_id,
        )


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class TTSRequest(BaseModel):
    text: str = Field(..., description="Văn bản tiếng Việt cần đọc (câu hỏi, đáp án...)")
    voice: Optional[str] = Field(None, description="Id giọng trong /v1/voices; bỏ trống = giọng mặc định")


def create_app(
    engine: Any = None,
    settings: Optional[Settings] = None,
    engine_loader: Callable[[Settings], Any] = load_engine,
) -> FastAPI:
    """Tạo app. Truyền ``engine`` (vd. engine giả trong test) để bỏ qua bước nạp model."""
    settings = settings or Settings.from_env()
    service = TTSService(settings)
    if engine is not None:
        service.attach(engine)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not service.ready:
            from starlette.concurrency import run_in_threadpool

            try:
                service.attach(await run_in_threadpool(engine_loader, settings))
            except Exception as exc:  # giữ server sống để /health báo lỗi rõ ràng
                service.load_error = f"{type(exc).__name__}: {exc}"
                logger.exception("Nạp model thất bại")
        yield

    app = FastAPI(title="VieNeu-TTS API (LearnAble)", version="1.0.0", lifespan=lifespan)
    app.state.service = service

    def require_api_key(x_api_key: Optional[str] = Header(None)) -> None:
        expected = settings.api_key
        if expected and not (x_api_key and hmac.compare_digest(x_api_key, expected)):
            raise HTTPException(status_code=401, detail="Thiếu hoặc sai X-API-Key")

    def require_ready() -> TTSService:
        if not service.ready:
            detail = f"Model lỗi khi nạp: {service.load_error}" if service.load_error else "Model đang nạp"
            raise HTTPException(status_code=503, detail=detail)
        return service

    @app.get("/health")
    def health() -> Response:
        body: dict[str, Any] = {
            "status": "ok" if service.ready else ("error" if service.load_error else "loading"),
            "backbone": settings.backbone,
            "lora": settings.lora,
            "model_version": service.model_version,
            "sample_rate": SAMPLE_RATE,
            "cache": settings.cache_dir is not None,
        }
        if service.ready:
            body["default_voice"] = service.default_voice()
        if service.load_error:
            body["error"] = service.load_error
        return Response(
            json.dumps(body, ensure_ascii=False),
            status_code=200 if service.ready else 503,
            media_type="application/json",
        )

    @app.get("/v1/voices", dependencies=[Depends(require_api_key)])
    def voices(svc: TTSService = Depends(require_ready)) -> dict[str, Any]:
        return {"default": svc.default_voice(), "voices": svc.voices()}

    @app.post(
        "/v1/tts",
        dependencies=[Depends(require_api_key)],
        response_class=Response,
        responses={200: {"content": {"audio/wav": {}}, "description": "WAV PCM16 mono 24 kHz"}},
    )
    def tts(req: TTSRequest, request: Request, svc: TTSService = Depends(require_ready)) -> Response:
        text = req.text.strip()
        if not text:
            raise HTTPException(status_code=422, detail="text không được rỗng")
        if len(text) > settings.max_chars:
            raise HTTPException(
                status_code=422, detail=f"text dài {len(text)} ký tự, tối đa {settings.max_chars}"
            )
        try:
            result = svc.synthesize(text, req.voice)
        except LookupError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Sinh audio lỗi")
            raise HTTPException(status_code=500, detail=f"Sinh audio lỗi: {type(exc).__name__}: {exc}") from exc
        logger.info(
            "tts voice=%s chars=%d cache=%s audio=%.2fs elapsed=%dms",
            result.voice, len(text), "HIT" if result.cache_hit else "MISS", result.duration_s, result.elapsed_ms,
        )
        return Response(
            content=result.wav,
            media_type="audio/wav",
            headers={
                "X-Cache": "HIT" if result.cache_hit else "MISS",
                "X-Cache-Key": result.cache_key,
                "X-Audio-Duration": f"{result.duration_s:.3f}",
                "X-Voice": result.voice,
                "X-Model-Version": svc.model_version,
                "X-Elapsed-Ms": str(result.elapsed_ms),
            },
        )

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings.from_env()
    # workers=1: mỗi worker nạp một bản model riêng.
    uvicorn.run(create_app(settings=settings), host=settings.host, port=settings.port, workers=1)


if __name__ == "__main__":
    main()
