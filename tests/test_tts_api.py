"""Tests cho apps/tts_api.py bằng engine giả (không cần torch/GPU)."""
import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from apps.tts_api import Settings, create_app, encode_wav


class FakeEngine:
    sample_rate = 24_000

    def __init__(self, fail: bool = False):
        self._default_voice = "ngoc_huyen"
        self._preset_voices = {
            "ngoc_huyen": {"codes": [1, 2, 3], "text": "Câu mẫu.", "description": "Giọng nữ"},
            "other": {"codes": [4, 5], "text": "Mẫu khác.", "description": "Giọng khác"},
        }
        self.calls = []
        self.fail = fail

    def list_preset_voices(self):
        return [(v["description"], k) for k, v in self._preset_voices.items()]

    def get_preset_voice(self, name=None):
        name = name or self._default_voice
        if name not in self._preset_voices:
            raise ValueError(f"Voice '{name}' not found")
        v = self._preset_voices[name]
        return {"codes": v["codes"], "text": v["text"]}

    def infer(self, text, voice=None, temperature=1.0, top_k=50, apply_watermark=True, **kwargs):
        if self.fail:
            raise RuntimeError("boom")
        self.calls.append({"text": text, "voice": voice, "temperature": temperature, "top_k": top_k})
        t = np.arange(int(0.5 * self.sample_rate)) / self.sample_rate
        return (1.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)  # vượt [-1, 1] để test clip


@pytest.fixture
def settings(tmp_path):
    return Settings(cache_dir=tmp_path / "cache", max_chars=50)


def make_client(settings, engine=None):
    return TestClient(create_app(engine=engine or FakeEngine(), settings=settings))


def test_health_reports_ready_model(settings):
    resp = make_client(settings).get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["default_voice"] == "ngoc_huyen"
    assert body["sample_rate"] == 24_000


def test_voices_lists_presets_and_default(settings):
    resp = make_client(settings).get("/v1/voices")
    assert resp.status_code == 200
    assert resp.json()["default"] == "ngoc_huyen"
    assert {v["id"] for v in resp.json()["voices"]} == {"ngoc_huyen", "other"}


def test_tts_returns_pcm16_wav_and_clips(settings):
    resp = make_client(settings).post("/v1/tts", json={"text": "Câu hỏi số một."})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert resp.content[:4] == b"RIFF"
    info = sf.info(io.BytesIO(resp.content))
    assert (info.samplerate, info.channels, info.subtype) == (24_000, 1, "PCM_16")
    audio, _ = sf.read(io.BytesIO(resp.content))
    assert np.abs(audio).max() <= 1.0
    assert float(resp.headers["X-Audio-Duration"]) == pytest.approx(0.5, abs=0.01)
    assert resp.headers["X-Voice"] == "ngoc_huyen"


def test_second_identical_request_is_cache_hit(settings):
    engine = FakeEngine()
    client = make_client(settings, engine)
    first = client.post("/v1/tts", json={"text": "Đáp án A: Hà Nội."})
    second = client.post("/v1/tts", json={"text": "  Đáp án A: Hà Nội.  "})
    assert (first.headers["X-Cache"], second.headers["X-Cache"]) == ("MISS", "HIT")
    assert first.content == second.content
    assert len(engine.calls) == 1


def test_cache_key_depends_on_text_and_voice(settings):
    client = make_client(settings)
    a = client.post("/v1/tts", json={"text": "Câu một."}).headers["X-Cache-Key"]
    b = client.post("/v1/tts", json={"text": "Câu hai."}).headers["X-Cache-Key"]
    c = client.post("/v1/tts", json={"text": "Câu một.", "voice": "other"}).headers["X-Cache-Key"]
    assert len({a, b, c}) == 3


def test_engine_receives_preset_voice_and_settings(settings):
    engine = FakeEngine()
    make_client(settings, engine).post("/v1/tts", json={"text": "Xin chào.", "voice": "other"})
    call = engine.calls[0]
    assert call["voice"] == {"codes": [4, 5], "text": "Mẫu khác."}
    assert (call["temperature"], call["top_k"]) == (settings.temperature, settings.top_k)


def test_cache_can_be_disabled(tmp_path):
    engine = FakeEngine()
    client = make_client(Settings(cache_dir=None), engine)
    client.post("/v1/tts", json={"text": "Câu một."})
    resp = client.post("/v1/tts", json={"text": "Câu một."})
    assert resp.headers["X-Cache"] == "MISS"
    assert len(engine.calls) == 2


@pytest.mark.parametrize("text", ["", "   ", "x" * 51])
def test_invalid_text_is_422(settings, text):
    assert make_client(settings).post("/v1/tts", json={"text": text}).status_code == 422


def test_unknown_voice_is_400(settings):
    resp = make_client(settings).post("/v1/tts", json={"text": "Câu một.", "voice": "nobody"})
    assert resp.status_code == 400


def test_engine_failure_is_500(settings):
    resp = make_client(settings, FakeEngine(fail=True)).post("/v1/tts", json={"text": "Câu một."})
    assert resp.status_code == 500
    assert "boom" in resp.json()["detail"]


def test_api_key_required_when_configured(tmp_path):
    client = make_client(Settings(cache_dir=None, api_key="secret-key"))
    assert client.post("/v1/tts", json={"text": "Câu một."}).status_code == 401
    assert client.get("/v1/voices", headers={"X-API-Key": "wrong"}).status_code == 401
    ok = client.post("/v1/tts", json={"text": "Câu một."}, headers={"X-API-Key": "secret-key"})
    assert ok.status_code == 200
    assert client.get("/health").status_code == 200  # health không cần key


def test_load_failure_reported_by_health(tmp_path):
    def broken_loader(_settings):
        raise RuntimeError("thiếu HF_TOKEN")

    app = create_app(settings=Settings(cache_dir=None), engine_loader=broken_loader)
    with TestClient(app) as client:  # context manager chạy lifespan -> gọi loader
        health = client.get("/health")
        assert health.status_code == 503
        assert "thiếu HF_TOKEN" in health.json()["error"]
        assert client.post("/v1/tts", json={"text": "Câu một."}).status_code == 503


def test_encode_wav_clips_out_of_range():
    data = encode_wav(np.array([2.0, -2.0, 0.5], dtype=np.float32), 24_000)
    audio, sr = sf.read(io.BytesIO(data))
    assert sr == 24_000
    assert audio.max() <= 1.0 and audio.min() >= -1.0
