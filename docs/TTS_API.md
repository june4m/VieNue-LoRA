# VieNeu-TTS API cho LearnAble

Service HTTP sinh audio câu hỏi và đáp án thi cho thí sinh khiếm thị. Code nằm ở [`apps/tts_api.py`](../apps/tts_api.py).

Khi khởi động, service nạp backbone PyTorch `pnnbao-ump/VieNeu-TTS-0.3B` cùng LoRA adapter đã finetune (xem [`finetune/kaggle/`](../finetune/kaggle/README.md)) một lần duy nhất. Sau đó nó nhận văn bản và trả về file WAV.

> [!IMPORTANT]
> Adapter chỉ chạy với backbone PyTorch 0.3B (`mode="standard"`), không chạy với model ONNX v3-Turbo mặc định của repo. Server cần `torch` + `peft` và nên có GPU.

## Endpoint

| Method | Đường dẫn | Mô tả |
|---|---|---|
| `GET` | `/health` | `200` khi model sẵn sàng. `503` khi đang nạp hoặc nạp lỗi (trường `error` ghi lý do). Không cần API key |
| `GET` | `/v1/voices` | `{"default": "ngoc_huyen", "voices": [{"id", "description"}]}` |
| `POST` | `/v1/tts` | Body `{"text": "...", "voice": "ngoc_huyen"}` (bỏ `voice` thì dùng giọng mặc định) → `audio/wav` PCM16 mono 24 kHz |

Header trả về của `/v1/tts`:

| Header | Ý nghĩa |
|---|---|
| `X-Cache` | `HIT` (lấy từ cache, không tốn GPU) hoặc `MISS` |
| `X-Cache-Key` | SHA-256 của model + giọng + text + tham số sinh. Backend dùng được làm tên file |
| `X-Audio-Duration` | Độ dài audio (giây) |
| `X-Voice`, `X-Model-Version` | Giọng đã dùng; backbone@hash của adapter |
| `X-Elapsed-Ms` | Thời gian xử lý phía server |

Mã lỗi: `401` sai hoặc thiếu `X-API-Key`; `422` text rỗng hoặc dài quá `VIENEU_MAX_CHARS`; `400` giọng không tồn tại; `503` model chưa sẵn sàng; `500` lỗi khi sinh audio.

```bash
curl -X POST http://localhost:8000/v1/tts -H "Content-Type: application/json" -H "X-API-Key: <key>" -d '{"text": "Câu hỏi số một. Thủ đô của nước Việt Nam là thành phố nào?"}' -o cau_01.wav
```

## Cấu hình (biến môi trường)

| Biến | Mặc định | Ghi chú |
|---|---|---|
| `VIENEU_LORA` | *(trống = base model)* | Thư mục `best_adapter` (có `adapter_model.safetensors` + `voices.json`) hoặc HF repo id |
| `VIENEU_BACKBONE` | `pnnbao-ump/VieNeu-TTS-0.3B` | Phải trùng backbone lúc train |
| `VIENEU_CODEC` | `neuphonic/neucodec` | Repo **gated**, cần `HF_TOKEN` hoặc cache sẵn + `HF_HUB_OFFLINE=1` |
| `HF_TOKEN` | — | Token Read của tài khoản HF đã đồng ý điều khoản codec |
| `VIENEU_DEVICE` / `VIENEU_CODEC_DEVICE` | `auto` / *(giống device)* | `cuda` hoặc `cpu`. Card ít VRAM thì để codec chạy `cpu` |
| `VIENEU_API_KEY` | *(trống = không kiểm tra)* | Bật khi service mở ra ngoài localhost |
| `VIENEU_CACHE_DIR` | `tts_cache` | `off` để tắt cache |
| `VIENEU_MAX_CHARS` | `2000` | Text dài được engine tự chia chunk 256 ký tự rồi nối lại |
| `VIENEU_TEMPERATURE` / `VIENEU_TOP_K` | `1.0` / `50` | Giống engine `standard` |
| `VIENEU_WATERMARK` | `true` | Watermark Perth (bỏ qua nếu chưa cài `perth`) |
| `VIENEU_EMOTION_TAG` | `false` | Để `false`: LoRA được train không có tag `<|emotion_0|>` |
| `VIENEU_HOST` / `VIENEU_PORT` | `0.0.0.0` / `8000` | |

Khi nạp adapter, service đối chiếu `r`/`lora_alpha` với `adapter_config.json`, và từ chối khởi động nếu lệch. Adapter được lưu bằng peft 0.21, còn lockfile của repo ghim peft 0.14.

## Chạy

**Máy dev (có GPU):**

```bash
uv sync --group gpu
```

```bash
VIENEU_LORA=finetune/output/kaggle/vieneu-lora/<run>/best_adapter HF_TOKEN=hf_xxx uv run python -m apps.tts_api
```

**Docker** (stage `tts-api` trong `docker/Dockerfile.gpu`). Chép thư mục `best_adapter` vào `models/lora/`, hoặc đặt `VIENEU_LORA_HOST_DIR`. Thêm `HF_TOKEN` và `VIENEU_API_KEY` vào `docker/.env`, rồi chạy:

```bash
docker compose -f docker/docker-compose.yml --profile tts-api up -d --build
```

Volume `huggingface_cache` giữ backbone và codec để không phải tải lại; volume `tts_cache` giữ audio đã sinh. Healthcheck chờ tối đa 10 phút cho lần nạp đầu.

Luôn chạy **một worker**: mỗi worker nạp một bản model riêng. Trong một process, các request sinh audio được xếp hàng chạy lần lượt; request trúng cache thì trả ngay.

> [!NOTE]
> **Windows:** import `pyarrow` sau `torch` có thể gây `access violation` (chuỗi `neucodec → torchtune → datasets → pyarrow`). Khi chạy trên Windows, import pyarrow trước:
>
> ```bash
> python -c "import pyarrow, pyarrow.dataset, runpy; runpy.run_module('apps.tts_api', run_name='__main__')"
> ```
>
> Linux/Docker không gặp lỗi này.

## Triển khai lên server riêng

> [!IMPORTANT]
> Service này chạy **PyTorch**, không phải ONNX. Engine ONNX trong repo (`apps/web_stream.py`, v3 Turbo) là model khác và không nạp được LoRA đã finetune. Muốn chạy chỉ bằng CPU (không GPU) thì phải merge LoRA vào base rồi chuyển sang GGUF + codec decoder ONNX. Hướng này chưa được làm.

Server khuyên dùng: Linux có GPU NVIDIA ≥ 8 GB VRAM, ví dụ AWS EC2 `g4dn.xlarge` (T4 16 GB), dùng AMI có sẵn driver NVIDIA + Docker.

1. Cài Docker, NVIDIA driver và **NVIDIA Container Toolkit**. Kiểm tra bằng `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi`.
2. Lấy code: `git clone https://github.com/june4m/VieNue-LoRA.git`, rồi checkout branch chứa `apps/tts_api.py`.
3. Chép adapter từ máy dev lên server:

   ```bash
   scp -r finetune/output/kaggle/vieneu-lora/<run>/best_adapter ubuntu@<server>:~/VieNue-LoRA/models/lora
   ```

4. Tạo `docker/.env` trên server, **không commit file này**:

   ```dotenv
   HF_TOKEN=hf_...            # token Read của tài khoản đã đồng ý điều khoản neuphonic/neucodec
   VIENEU_API_KEY=<chuỗi ngẫu nhiên dài>
   TTS_API_PORT=8000
   ```

5. Build và chạy:

   ```bash
   docker compose -f docker/docker-compose.yml --profile tts-api up -d --build
   ```

   Lần đầu sẽ tải backbone + codec (vài GB). Theo dõi bằng `docker logs -f vieneu-tts-api` cho tới khi `curl http://localhost:8000/health` trả `"status": "ok"`.
6. **Mạng:** chỉ mở port 8000 cho server backend LearnAble (security group hoặc firewall theo IP/private network), không mở ra Internet. Backend cấu hình `Tts:BaseUrl=http://<ip-private>:8000` và `Tts:ApiKey` bằng đúng `VIENEU_API_KEY`.
7. **Cập nhật model:** chép adapter mới vào `models/lora`, chạy `docker compose ... restart tts-api`. Cache key chứa hash của adapter, nên audio cũ không bị dùng nhầm cho model mới.

## Tích hợp với backend LearnAble (.NET)

Câu hỏi và đáp án là văn bản cố định, nên **sinh audio sẵn khi giảng viên công bố đề**, không sinh lúc thí sinh làm bài:

1. Khi công bố đề, backend gọi `POST /v1/tts` cho từng câu hỏi và đáp án, rồi lưu WAV vào storage (đặt tên theo `X-Cache-Key`) cùng đường dẫn trong DB.
2. Khi thí sinh làm bài, backend chỉ phát file đã lưu. Không phải chờ GPU, không quá tải giữa giờ thi.
3. Sửa nội dung câu hỏi sẽ ra text mới, tức key mới, nên chỉ phải sinh lại đúng câu đã sửa.

```csharp
public sealed class TtsClient(HttpClient http)
{
    public async Task<(byte[] Wav, string Key)> SynthesizeAsync(string text, string? voice = null, CancellationToken ct = default)
    {
        using var resp = await http.PostAsJsonAsync("/v1/tts", new { text, voice }, ct);
        resp.EnsureSuccessStatusCode();
        var key = resp.Headers.GetValues("X-Cache-Key").First();
        return (await resp.Content.ReadAsByteArrayAsync(ct), key);
    }
}

// Program.cs
builder.Services.AddHttpClient<TtsClient>(c =>
{
    c.BaseAddress = new Uri(builder.Configuration["Tts:BaseUrl"]!);
    c.DefaultRequestHeaders.Add("X-API-Key", builder.Configuration["Tts:ApiKey"]);
    c.Timeout = TimeSpan.FromMinutes(2); // câu dài có thể mất vài chục giây trên CPU
});
```

## Test

```bash
uv run pytest tests/test_tts_api.py
```

Test dùng engine giả nên không cần torch hay GPU. Phạm vi: health, voices, header WAV, cache HIT/MISS, API key, các mã lỗi 400/422/500/503.

## Kiểm chứng với adapter thật (08/11/2026)

Adapter `ngoc_huyen-full-20261108-1234-s42`, chạy trên laptop RTX 3050 Ti 4 GB (backbone GPU, codec CPU, máy thiếu RAM):

| Request | Ký tự | Audio | Thời gian |
|---|---:|---:|---:|
| Câu hỏi ngắn | 58 | 3,8 s | 45 s (MISS) / 19 ms (HIT, giống từng byte) |
| Câu hỏi + 4 đáp án (qua đường chia chunk + batch) | 306 | 19,9 s | 174 s |
| "Kỳ thi THPT quốc gia năm 2025 có 4 môn…" | 86 | 7,3 s | 93 s |

`/health` báo đã nạp LoRA, `default_voice=ngoc_huyen`, `emotion_tag=None`; kiểm tra `r`/`alpha` qua. Tất cả audio là PCM16 24 kHz, có tiếng.

Số liệu trên laptop không đại diện cho server. Trên Kaggle T4 (fp32, codec GPU), mỗi câu 4–8 s audio mất khoảng 12–24 s. Đề 40 câu × 5 đoạn (câu hỏi + 4 đáp án) khoảng 200 đoạn, tức cỡ 40–60 phút sinh sẵn một lần khi công bố đề. Tăng tốc (fp16 trên GPU, gom nhiều câu vào một batch) là việc có thể làm sau.
