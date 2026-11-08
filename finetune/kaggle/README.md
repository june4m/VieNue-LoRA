# Fine-tune VieNeu-TTS-0.3B (LoRA) trên Kaggle

Thư mục này đưa pipeline LoRA của `finetune/` lên Kaggle GPU (T4 16 GB, 30 giờ/tuần) thay cho Colab.

| File | Vai trò |
|---|---|
| `vieneu_kaggle_train.py` | Kernel chạy trọn gói: cài dependency → lấy repo ở commit cố định → tải + lọc + encode dataset → train LoRA → sinh audio mẫu → ghi adapter + `voices.json` |
| `kernel-metadata.json` | Cấu hình kernel (private, GPU T4, bật Internet) cho `kaggle kernels push` |

> [!IMPORTANT]
> Adapter tạo ra chỉ dùng được với backbone PyTorch `pnnbao-ump/VieNeu-TTS-0.3B` (`Vieneu(mode="standard")`), không dùng được với model ONNX v3-Turbo mặc định. Server API phải có `torch` + `peft` (nên có GPU).

## Yêu cầu một lần

1. Cài CLI: `uv tool install --python 3.12 kaggle --with six`
2. Đăng nhập: `kaggle auth login` (xác nhận OAuth trên trình duyệt).
3. Tài khoản Kaggle phải **xác minh số điện thoại** thì kernel mới bật được GPU và Internet.
4. Codec `neuphonic/neucodec` là repo **gated** trên Hugging Face:
   - bấm đồng ý điều khoản tại https://huggingface.co/neuphonic/neucodec;
   - tạo token **Read** tại https://huggingface.co/settings/tokens;
   - trên Kaggle mở notebook → **Add-ons → Secrets** → thêm `HF_TOKEN` và bật cho notebook.
5. Đổi `id` trong `kernel-metadata.json` nếu dùng tài khoản khác.

## Quy trình

> [!NOTE]
> Bản chạy do `kaggle kernels push` kích hoạt **không đọc được Kaggle Secrets** (`ConnectionError`). Vì vậy chỉ dùng CLI để lưu code (`--no-run`), còn việc chạy thì bấm trên web để có `HF_TOKEN`.

```bash
# 1. Lưu code thành version mới, không chạy
kaggle kernels push -p finetune/kaggle --accelerator NvidiaTeslaT4 --no-run
```

2. Trên Kaggle: mở notebook → **Edit** → tải lại trang (F5) → Ctrl+F kiểm tra dòng `RUN_MODE = "..."` đúng bản vừa push → **Settings**: GPU T4 x2, Internet On → **Save Version → Save & Run All (Commit)**. Có thể đóng tab, kernel vẫn chạy trên server.

```bash
# 3. Theo dõi
kaggle kernels status nguyencongminhk18hcm/vieneu-tts-lora-finetune
kaggle kernels logs   nguyencongminhk18hcm/vieneu-tts-lora-finetune

# 4. Tải kết quả về máy (thư mục finetune/output/ đã được gitignore)
kaggle kernels output nguyencongminhk18hcm/vieneu-tts-lora-finetune -p finetune/output/kaggle
```

| `RUN_MODE` | Dữ liệu | Train | Thời gian T4 (đã đo 08/11/2026) |
|---|---|---|---|
| `smoke` | 160 clip từ shard đầu | 20 step, batch 4 | 6,7 phút |
| `full` | 3.975 clip 3–15 s qua bộ lọc (9,6 giờ audio) | 1.416 step = 3 epoch, batch 4 × accum 2 | 108,5 phút (encode 41,7 + train 61,9), VRAM đỉnh 4,92 GiB, ~2,5 giờ quota GPU |

Kết quả full run đầu tiên: eval_loss 4,92 → 4,24, đi ngang ở 3 lần eval cuối (chênh < 0,006); train loss ~4,14, không có dấu hiệu overfit. Checkpoint tốt nhất là checkpoint cuối, nên train thêm epoch chỉ cải thiện rất ít.

Script đang để `RUN_MODE = "full"`.

Dòng log đầu tiên in `RUN_MODE=...`; nếu không khớp với bản vừa push thì editor đang giữ bản nháp cũ.

### Kết quả trong `/kaggle/working`

```text
data/metadata_encoded.csv          # filename|text|codes — dùng lại cho lần train sau
data/encode_info.json
vieneu-lora/<run_name>/
├── best_adapter/                  # LoRA adapter (checkpoint eval_loss tốt nhất) + tokenizer + voices.json
├── checkpoints/                   # 2 checkpoint gần nhất (để resume)
├── samples/NN_base.wav, NN_lora.wav  # cùng câu hỏi/đáp án: base vs fine-tuned
├── split.json                     # danh sách train/validation theo seed
├── log_history.json               # train/eval loss
└── run_note.json                  # môi trường, siêu tham số, thống kê dữ liệu, kết quả
```

### Dùng lại dữ liệu đã encode / resume

Gắn output của lần chạy trước làm input bằng cách thêm vào `kernel-metadata.json`:

```json
"kernel_sources": ["nguyencongminhk18hcm/vieneu-tts-lora-finetune"]
```

Script tự tìm `metadata_encoded.csv` trong `/kaggle/input` (nếu `REUSE_ENCODED = True`) và bỏ qua bước encode. Dữ liệu chỉ được dùng lại khi `encode_info.json` khớp cả dataset, codec, shard và `max_samples` của profile hiện tại, nên run `full` không vô tình lấy dữ liệu 160 clip của `smoke`. Để tiếp tục một run bị dừng, đặt `RESUME_FROM` thành đường dẫn checkpoint trong `/kaggle/input/...`.

## Xử lý lỗi thường gặp

| Triệu chứng trong log | Nguyên nhân / cách sửa |
|---|---|
| `Preflight` báo không có GPU/Internet, `torch ...+cpu`, `Failed to resolve 'pypi.org'` | Tài khoản chưa xác minh số điện thoại nên Kaggle tắt GPU + Internet dù metadata bật. Xác minh tại kaggle.com/settings rồi push lại |
| `Thiếu HF_TOKEN` / `GatedRepoError` / `chưa có quyền vào neuphonic/neucodec` | Làm bước 4 ở trên; token phải thuộc tài khoản HF đã bấm đồng ý |
| `No module named 'torchao.dtypes.nf4tensor'` | `torchtune` 0.6.1 không chạy với `torchao` ≥ 0.18; script đã ghim `torchao==0.17.0` |
| `pip đã thay đổi bộ torch` | Một dependency đòi torch khác bản của image; cập nhật image hoặc ghim version dependency đó |
| CUDA OOM khi train | Giảm `batch_size` và tăng `grad_accum` trong `PROFILES` để giữ nguyên effective batch |

## Khác biệt so với notebook Colab

- **fp16 thay cho bf16**: T4 (compute 7.5) không hỗ trợ bf16. Script tự chọn bf16 nếu GPU có hỗ trợ.
- **Một GPU**: đặt `CUDA_VISIBLE_DEVICES=0` để tránh DataParallel trên T4 x2.
- **Không cài lại torch**: chỉ cài `peft`, `neucodec`, `sea-g2p`, và khóa phiên bản torch có sẵn bằng constraints.
- **Padding động + `group_by_length`** thay vì pad cố định 2048 token. Mẫu dài quá 2048 token bị loại, không bị cắt, để model luôn học được token kết thúc.
- **Label không tính pad**, validation split có seed, chọn checkpoint theo `eval_loss`, tự dừng và lưu trước giới hạn 12 giờ.
- **Phonemizer lấy từ repo** (`vieneu_utils.phonemize_text`) ở commit cố định, nên khớp với engine của API.

## Dùng adapter trong API

Service HTTP đóng gói sẵn adapter: xem [docs/TTS_API.md](../../docs/TTS_API.md) (`apps/tts_api.py`). Nếu muốn gọi engine trực tiếp, **hai tham số dưới là bắt buộc**:

```python
from vieneu import Vieneu

tts = Vieneu(mode="standard", backbone_repo="pnnbao-ump/VieNeu-TTS-0.3B",
             backbone_device="cuda", codec_repo="neuphonic/neucodec", codec_device="cuda",
             gguf_filename=None,   # mặc định là file GGUF -> không nạp được LoRA
             emotion="none")       # LoRA train KHÔNG có tag <|emotion_0|> mà emotion="natural" chèn vào
tts.load_lora_adapter("finetune/output/kaggle/vieneu-lora/<run_name>/best_adapter")  # đọc luôn voices.json
audio = tts.infer("Câu hỏi số một. Thủ đô của nước Việt Nam là thành phố nào?")
tts.save(audio, "question_01.wav")
```

Máy chạy cần `torch`, `transformers`, `peft`, `neucodec` (nếu cài `neucodec` mới thì ghim `torchao==0.17.0`), và quyền vào repo gated `neuphonic/neucodec`: biến môi trường `HF_TOKEN`, hoặc cache sẵn weights rồi đặt `HF_HUB_OFFLINE=1`.

Dataset `pnnbao-ump/ngochuyen_voice` có license **CC BY-NC 4.0**, chỉ dùng phi thương mại (đồ án được).
