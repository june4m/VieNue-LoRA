# WBS — Fine-tune VieNeu-TTS 0.3B bằng LoRA

Tài liệu này phân rã `FINETUNING_PLAN.vi.md` thành các công việc nhỏ, có thể giao, theo dõi và nghiệm thu độc lập.

## 1. Quy ước

- **P**: người phụ trách dự án/data; **ML**: kỹ sư ML; **QA**: người kiểm thử dữ liệu/audio; **RV**: người nghe đánh giá.
- Ước lượng là **giờ công**, không bao gồm thời gian chờ thu âm, tải model hoặc chạy GPU, trừ khi có ghi riêng.
- Trạng thái đề xuất: `TODO`, `DOING`, `BLOCKED`, `DONE`.
- Một task chỉ được đánh dấu `DONE` khi đầu ra và tiêu chí nghiệm thu trong bảng đều đạt.
- Backbone bắt buộc của WBS này là `pnnbao-ump/VieNeu-TTS-0.3B`; không áp dụng cho model ONNX `VieNeu-TTS-v3-Turbo`.

## 2. Mốc và cổng nghiệm thu

| Mốc | Kết quả bắt buộc |
|---|---|
| M0 — Phạm vi hợp lệ | Có mục tiêu, quyền sử dụng, tiêu chí thành công và test set khóa kín |
| M1 — Dữ liệu sẵn sàng | Audio sạch, metadata hợp lệ, split cố định và không rò rỉ |
| M2 — Pipeline sẵn sàng | Filter/encode tái lập được; train có validation, seed và kiểm soát truncation |
| M3 — Smoke test đạt | LoRA train/save/load/inference được từ đầu đến cuối |
| M4 — Có ứng viên | Full run hoàn tất và có checkpoint được chọn bằng validation + nghe thử |
| M5 — Sẵn sàng phát hành | Adapter/merged model đạt test khóa kín, đủ tài liệu và có bản rollback |

## 3. WBS chi tiết

### 0. Quản trị phạm vi và quyền sử dụng

| ID | Công việc | Phụ trách | Ước lượng | Phụ thuộc | Đầu ra / tiêu chí nghiệm thu |
|---|---|---:|---:|---|---|
| 0.1 | Chốt use case và người dùng mục tiêu | P | 0.5h | — | Mô tả ngắn mục đích dùng model và phạm vi phân phối |
| 0.2 | Xác nhận chỉ dùng một người nói | P | 0.25h | 0.1 | Speaker mục tiêu được định danh; không trộn speaker |
| 0.3 | Thu văn bản đồng ý của người nói | P | 1h | 0.1 | Consent nêu rõ quyền train, lưu trữ và phát hành |
| 0.4 | Kiểm tra giấy phép base model | P | 0.5h | 0.1 | Kết luận license phù hợp use case, có đường dẫn/phiên bản |
| 0.5 | Kiểm tra giấy phép và nguồn dataset | P | 0.5h | 0.2 | Mọi audio có nguồn và quyền sử dụng rõ ràng |
| 0.6 | Chốt ngưỡng thành công | P, ML, QA | 0.5h | 0.1 | Ghi đủ: ưu tiên giọng ≥70%, lỗi từ ≤1/100 từ, ổn định ≥95%, test khóa kín |
| 0.7 | Tạo quy ước đặt tên run/checkpoint | ML | 0.25h | — | Ví dụ: `<speaker>-<stage>-<yyyymmdd>-s<seed>` |
| 0.8 | Xác minh dữ liệu/checkpoint được Git ignore | ML | 0.5h | — | `finetune/dataset/` và `finetune/output/` không thể bị add nhầm |

**Gate M0-A:** Chỉ thu thập/huấn luyện khi 0.3–0.6 đều `DONE`.

### 1. Tạo tập đánh giá và baseline

| ID | Công việc | Phụ trách | Ước lượng | Phụ thuộc | Đầu ra / tiêu chí nghiệm thu |
|---|---|---:|---:|---|---|
| 1.1 | Tạo thư mục `finetune/eval/` | ML | 0.25h | — | Có `base/`, `checkpoints/` và vị trí lưu scorecard |
| 1.2 | Soạn 30–50 prompt test | QA | 2h | 0.6 | Prompt tự nhiên, không có lỗi chính tả |
| 1.3 | Gắn category cho từng prompt | QA | 0.5h | 1.2 | Có câu kể, hỏi, cảm thán, tên riêng, từ vay mượn, ngày tháng, câu ngắn/dài |
| 1.4 | Đánh dấu tối thiểu 20 prompt khóa kín | QA | 0.25h | 1.2 | 20+ câu cam kết không xuất hiện trong train/checkpoint selection |
| 1.5 | Tạo `eval/prompts.csv` | QA | 0.5h | 1.3, 1.4 | Mỗi dòng parse được theo `id|category|text`; ID duy nhất |
| 1.6 | Chốt cấu hình inference baseline | ML | 0.5h | 1.5 | Ghi revision model/tokenizer, seed và tham số inference |
| 1.7 | Sinh audio từ base model | ML | 1h + GPU | 1.6 | Mỗi prompt có đúng một audio trong `eval/base/`; không lỗi sinh |
| 1.8 | Tạo mẫu `eval/scorecard.csv` | QA | 0.5h | 1.5 | Có sample ID, system ẩn danh, reviewer, similarity, intelligibility, naturalness, stability, prosody, lỗi từ, ghi chú |
| 1.9 | Chấm baseline | RV (≥2) | 2h/người | 1.7, 1.8 | Tất cả mẫu được ít nhất hai người chấm độc lập |
| 1.10 | Khóa và lưu hash test manifest | ML | 0.5h | 1.5 | Hash được ghi lại; file được xem là bất biến trong dự án |

**Gate M0-B:** Baseline, prompt khóa kín, config inference và scorecard đã tồn tại.

### 2. Chuẩn bị môi trường tái lập

| ID | Công việc | Phụ trách | Ước lượng | Phụ thuộc | Đầu ra / tiêu chí nghiệm thu |
|---|---|---:|---:|---|---|
| 2.1 | Đồng bộ dependency nhóm GPU | ML | 0.5h | — | `uv sync --group gpu` thành công |
| 2.2 | Cài và pin PEFT | ML | 0.5h | 2.1 | `import peft` thành công; version được ghi lại |
| 2.3 | Kiểm tra PyTorch nhận CUDA | ML | 0.25h | 2.1 | `torch.cuda.is_available()` trả về `True` |
| 2.4 | Kiểm tra GPU và VRAM | ML | 0.25h | 2.3 | Ghi model GPU; khuyến nghị VRAM ≥12 GB |
| 2.5 | Kiểm tra hỗ trợ BF16 | ML | 0.25h | 2.3 | Có kết luận dùng BF16 hay cần smoke test FP16 |
| 2.6 | Lưu snapshot môi trường | ML | 0.5h | 2.2–2.5 | Run note có `nvidia-smi`, Python version và `uv pip freeze` |
| 2.7 | Ghi revision base model/tokenizer/codec | ML | 0.5h | 2.2 | Mỗi dependency từ Hugging Face có repo và revision xác định |

**Gate:** Nếu 2.3 không đạt, dừng huấn luyện; CPU không thuộc phạm vi WBS.

### 3. Thu thập và chuẩn hóa dữ liệu

| ID | Công việc | Phụ trách | Ước lượng | Phụ thuộc | Đầu ra / tiêu chí nghiệm thu |
|---|---|---:|---:|---|---|
| 3.1 | Chuẩn bị phòng, micro và mức gain cố định | P, QA | 1h | 0.2 | Có đoạn test không clipping, vang, nhạc nền hoặc nhiễu đáng kể |
| 3.2 | Chuẩn bị kịch bản thu cân bằng | QA | 3h | 0.2 | Phủ âm vị, thanh điệu, dấu câu, độ dài và mục đích diễn đạt |
| 3.3 | Loại nội dung trùng prompt khóa kín | QA | 0.5h | 1.10, 3.2 | Không có câu khóa kín trong kịch bản thu |
| 3.4 | Thu âm người nói | P, QA | Theo dữ liệu | 3.1–3.3 | Mục tiêu 2–4 giờ raw audio; pilot tối thiểu 30–60 phút |
| 3.5 | Cắt audio theo ranh giới tự nhiên | QA | 1.5x thời lượng audio | 3.4 | Mỗi clip 3–15 giây; không cắt giữa từ/nhịp thở |
| 3.6 | Đặt tên file duy nhất | QA | 0.5h | 3.5 | Theo mẫu `audio_0001.wav`; không trùng tên |
| 3.7 | Chuẩn hóa WAV mono 16 kHz | ML | 1h | 3.5 | Công cụ kiểm tra xác nhận toàn bộ file đúng format |
| 3.8 | Chép lời nguyên văn từng clip | QA | 4–6x thời lượng audio | 3.5 | Transcript khớp lời nói, gồm dấu câu; không tự sửa lỗi nói |
| 3.9 | Chuẩn hóa số và chữ viết tắt theo lời đọc | QA | 1h | 3.8 | Transcript không còn chữ số/acronym viết hoa ngoài chủ đích |
| 3.10 | Loại ký tự phân cách `|` khỏi transcript | QA | 0.25h | 3.8 | Không transcript nào chứa `|` |
| 3.11 | Tạo `dataset/metadata.csv` | QA | 0.5h | 3.6, 3.8–3.10 | Mọi dòng đúng `filename|text`; filename tồn tại |
| 3.12 | Quét file thiếu/không đọc được | ML | 0.5h | 3.7, 3.11 | Số file thiếu = 0; số file không đọc được = 0 |
| 3.13 | Phát hiện clip trùng/gần trùng | ML | 1h | 3.11 | Danh sách nghi trùng được tạo và review |
| 3.14 | Loại clip trùng đã xác nhận | QA | 0.5h | 3.13 | Không còn nội dung trùng gây overweight |
| 3.15 | Phát hiện clipping, silence, duration bất thường | ML | 1h | 3.7 | Có report tự động và danh sách clip cần nghe |
| 3.16 | Nghe 100% clip bị cảnh báo | QA | Theo số lỗi | 3.15 | Mỗi cảnh báo có quyết định giữ/loại/sửa |
| 3.17 | Nghe mẫu ngẫu nhiên ≥10% clip còn lại | QA | Theo dữ liệu | 3.11 | Tỷ lệ transcript mismatch trong mẫu <1% |
| 3.18 | Loại clip nhiễu/vang/chồng giọng/vỡ tiếng | QA | 1h | 3.16, 3.17 | Chỉ audio được chấp nhận còn trong cleaned manifest |
| 3.19 | Tính số clip và tổng thời lượng accepted | ML | 0.25h | 3.18 | Report có count, duration tổng và phân bố duration |

### 4. Chia train/validation/test

| ID | Công việc | Phụ trách | Ước lượng | Phụ thuộc | Đầu ra / tiêu chí nghiệm thu |
|---|---|---:|---:|---|---|
| 4.1 | Gắn session/source ID cho clip | QA | 0.5h | 3.18 | Mỗi clip truy được về buổi/file nguồn |
| 4.2 | Chọn seed chia tập cố định | ML | 0.25h | 4.1 | Seed được ghi trong run config |
| 4.3 | Chia theo session ở tỷ lệ 90/5/5 | ML | 0.5h | 4.1, 4.2 | Có manifest train, validation, test |
| 4.4 | Kiểm tra session không xuyên tập | ML | 0.25h | 4.3 | Một session chỉ thuộc một split |
| 4.5 | Kiểm tra text trùng giữa các split | ML | 0.5h | 4.3 | Không có exact duplicate; near-duplicate được review |
| 4.6 | Kiểm tra giao với prompt khóa kín | ML | 0.5h | 1.10, 4.3 | Không có prompt khóa kín trong bất kỳ split train nào |
| 4.7 | Khóa manifest và ghi hash | ML | 0.5h | 4.4–4.6 | Split tái tạo được từ seed/session; hash được lưu |

**Gate M1:** 3.12, 3.17–3.19 và 4.4–4.7 đều đạt.

### 5. Lọc và encode dữ liệu

| ID | Công việc | Phụ trách | Ước lượng | Phụ thuộc | Đầu ra / tiêu chí nghiệm thu |
|---|---|---:|---:|---|---|
| 5.1 | Chạy `filter_data.py` trên manifest accepted | ML | 0.5h | M1 | `metadata_cleaned.csv` được tạo không lỗi |
| 5.2 | Tổng hợp lý do bị loại | ML | 0.5h | 5.1 | Có count cho duration, chữ số, acronym, thiếu dấu câu và lỗi khác |
| 5.3 | Nghe mẫu mỗi nhóm bị loại | QA | 0.5h | 5.2 | Xác nhận filter không loại sai hàng loạt |
| 5.4 | Sửa transcript hợp lệ bị reject | QA | 1h | 5.3 | Số/acronym được viết theo lời đọc; không xóa nội dung tùy tiện |
| 5.5 | Chạy lại filter và chốt cleaned manifest | ML | 0.5h | 5.4 | Tất cả dòng còn lại thỏa rule 3–15 giây và transcript |
| 5.6 | Thêm tham số `--seed` cho `encode_data.py` | ML | 1h | 5.5 | Hai lần chạy cùng seed cho cùng thứ tự mẫu |
| 5.7 | Thêm tham số `--max_samples` | ML | 1h | 5.5 | Pilot giới hạn được; production encode được toàn bộ train |
| 5.8 | Ngăn ghi đè output ngoài chủ đích | ML | 0.5h | 5.5 | Script yêu cầu output path hoặc xác nhận/flag overwrite rõ ràng |
| 5.9 | Encode riêng train manifest | ML | 1h + GPU | 5.6–5.8 | `metadata_encoded.csv` chỉ chứa train clip |
| 5.10 | Đối soát số dòng encode | ML | 0.25h | 5.9 | Số encoded = số train dự kiến (hoặc đúng giới hạn pilot) |
| 5.11 | Parse kiểm tra từng dòng encoded | ML | 0.5h | 5.9 | 100% dòng đúng `filename|text|JSON-array` |
| 5.12 | Kiểm tra code array | ML | 0.5h | 5.11 | Không array rỗng; tất cả token trong miền codec hợp lệ |
| 5.13 | Kiểm tra không lẫn validation/test | ML | 0.25h | 5.9, 4.7 | Giao filename với validation/test bằng 0 |
| 5.14 | Lưu codec repo/revision và hash output | ML | 0.25h | 5.12 | Encode artifact truy vết và tái lập được |

### 6. Nâng cấp pipeline training để đo lường được

| ID | Công việc | Phụ trách | Ước lượng | Phụ thuộc | Đầu ra / tiêu chí nghiệm thu |
|---|---|---:|---:|---|---|
| 6.1 | Bổ sung loader train/validation riêng | ML | 2h | 4.7, 5.9 | `train.py` nhận đúng hai dataset cố định |
| 6.2 | Truyền `eval_dataset` vào Trainer | ML | 0.5h | 6.1 | Evaluation chạy theo lịch cấu hình |
| 6.3 | Bật log validation loss | ML | 0.5h | 6.2 | Run log/checkpoint có `eval_loss` |
| 6.4 | Thêm seed cho Python | ML | 0.25h | — | `random` cho kết quả tái lập |
| 6.5 | Thêm seed cho PyTorch/CUDA | ML | 0.25h | — | Seed được set trước khởi tạo model/dataloader |
| 6.6 | Thêm seed vào `TrainingArguments` | ML | 0.25h | — | Seed và data seed hiện trong run config |
| 6.7 | Thêm bộ đếm sequence bị truncate | ML | 1h | 5.9 | Report số lượng/tỷ lệ mẫu vượt 2.048 token |
| 6.8 | Xử lý mẫu vượt giới hạn | ML, QA | 1h | 6.7 | Mẫu được loại/chia lại; hoặc tăng limit có benchmark VRAM |
| 6.9 | Xác minh `use_4bit` có thực sự được dùng | ML | 0.5h | — | Config không tuyên bố QLoRA sai; hoặc model loading dùng 4-bit đúng cách |
| 6.10 | Cấu hình precision theo GPU | ML | 0.5h | 2.5 | BF16 khi hỗ trợ; FP16 chỉ sau test loss ổn định |
| 6.11 | Thêm resume-from-checkpoint | ML | 1h | — | Run gián đoạn có thể tiếp tục cùng run/config |
| 6.12 | Log số tham số trainable | ML | 0.5h | — | Log chứng minh chỉ adapter LoRA được cập nhật |
| 6.13 | Log đầy đủ metadata của run | ML | 1h | 0.7, 2.6, 2.7, 4.7, 5.14 | Run note có commit, revision, dataset hash, seed, hyperparameter, GPU/VRAM |
| 6.14 | Tính step theo 3–5 epoch | ML | 0.5h | 5.10 | `max_steps` có phép tính từ sample count/effective batch size |
| 6.15 | Thêm lịch lưu checkpoint đủ dày | ML | 0.5h | 6.14 | Có checkpoint đầu/giữa/cuối để so sánh |
| 6.16 | Test loader và sequence builder | ML | 1.5h | 6.1, 6.7 | Test phủ split đúng, parse lỗi và truncate counting |

**Gate M2:** Training có validation, seed cố định, truncation được kiểm soát, resume được và run truy vết đầy đủ.

### 7. Thử nghiệm 0 — smoke test

| ID | Công việc | Phụ trách | Ước lượng | Phụ thuộc | Đầu ra / tiêu chí nghiệm thu |
|---|---|---:|---:|---|---|
| 7.1 | Chọn 50–100 clip sạch | QA | 0.5h | M1 | Smoke manifest cố định, không chứa validation/test |
| 7.2 | Tạo run config 20–50 step | ML | 0.25h | M2, 7.1 | Run name riêng; LoRA r=16, alpha=32, dropout=0.05, LR=2e-4 |
| 7.3 | Chạy smoke training | ML | 0.5h + GPU | 7.2 | Process kết thúc thành công |
| 7.4 | Kiểm tra loss hữu hạn | ML | 0.25h | 7.3 | Không NaN/Inf; xu hướng loss hợp lý |
| 7.5 | Kiểm tra chỉ LoRA được update | ML | 0.25h | 7.3 | Base weights frozen; trainable params đúng mục tiêu |
| 7.6 | Kiểm tra checkpoint được lưu | ML | 0.25h | 7.3 | Adapter/config tồn tại và đọc được |
| 7.7 | Load adapter vào pipeline inference | ML | 0.5h | 7.6 | Load không thiếu/mismatch key |
| 7.8 | Sinh và lưu audio test | ML, QA | 0.5h | 7.7 | Audio phát được, không im lặng/noise collapse |
| 7.9 | Test resume từ checkpoint | ML | 0.5h + GPU | 7.6 | Step tiếp tục đúng, không tạo run mất truy vết |
| 7.10 | Ghi kết luận smoke test | ML | 0.25h | 7.4–7.9 | Pass/fail và lỗi còn lại được ghi rõ |

**Gate M3:** 7.4–7.9 đều đạt. Không dùng smoke test để kết luận chất lượng giọng.

### 8. Thử nghiệm 1 — pilot

| ID | Công việc | Phụ trách | Ước lượng | Phụ thuộc | Đầu ra / tiêu chí nghiệm thu |
|---|---|---:|---:|---|---|
| 8.1 | Chọn 30–60 phút audio sạch nhất | QA | 1h | M3 | Pilot manifest và hash được lưu |
| 8.2 | Tính step budget pilot | ML | 0.25h | 8.1 | Ghi rõ số epoch, effective batch size và max steps |
| 8.3 | Chốt config pilot | ML | 0.25h | 8.2 | r=16, alpha=32, dropout=0.05, LR=2e-4; seed cố định |
| 8.4 | Chạy pilot | ML | 0.5h + GPU | 8.3 | Run hoàn tất hoặc resume thành công |
| 8.5 | Chọn checkpoint đầu/giữa/cuối | ML | 0.25h | 8.4 | Ít nhất 3 checkpoint để so sánh |
| 8.6 | Sinh cùng tập prompt cho 3 checkpoint | ML | 1h + GPU | 8.5 | Audio dùng cùng prompt/config inference |
| 8.7 | Nghe kiểm tra nhanh | QA, RV | 1h/người | 8.6 | Có ghi nhận similarity, sai phát âm, lặp và overfitting |
| 8.8 | Chọn config cho full run | ML, QA | 0.5h | 8.7 | Quyết định dựa trên eval loss + nghe thử, có lý do |

### 9. Thử nghiệm 2 — full data

| ID | Công việc | Phụ trách | Ước lượng | Phụ thuộc | Đầu ra / tiêu chí nghiệm thu |
|---|---|---:|---:|---|---|
| 9.1 | Encode toàn bộ train accepted | ML | 1h + GPU | 5.6–5.14, 8.8 | Count khớp train manifest; output có hash |
| 9.2 | Tính ngân sách 3–5 epoch | ML | 0.25h | 9.1 | Max steps tính từ mẫu, batch, accumulation và GPU count |
| 9.3 | Chốt full-run config | ML | 0.25h | 8.8, 9.2 | Mọi hyperparameter và seed được lưu trước khi chạy |
| 9.4 | Chạy full training | ML | 0.5h + GPU | 9.3 | Run hoàn tất hoặc resume đúng checkpoint |
| 9.5 | Theo dõi train/eval loss và VRAM | ML | Trong khi chạy | 9.4 | Không NaN/Inf/OOM; peak VRAM và curve được lưu |
| 9.6 | Lập shortlist ≥3 checkpoint | ML | 0.5h | 9.4, 9.5 | Gồm các vùng validation tốt, không mặc định lấy checkpoint cuối |
| 9.7 | Sinh audio đánh giá cho shortlist | ML | 1h + GPU | 9.6 | Mọi hệ thống dùng đúng prompt và inference config |
| 9.8 | Ẩn danh và randomize tên file | QA | 0.5h | 9.7 | Reviewer không suy ra checkpoint từ filename/thứ tự |
| 9.9 | Hai người chấm toàn bộ mẫu | RV (≥2) | Theo bộ test | 9.8 | Mỗi sample có đủ 5 tiêu chí và lỗi từ |
| 9.10 | Tính metric theo toàn bộ và category khó | QA | 1h | 9.9 | Có tổng điểm và riêng tên riêng/câu dài/câu hỏi/... |
| 9.11 | Kiểm tra overfitting | ML, QA | 1h | 9.5, 9.10 | Kiểm tra divergence loss, copy câu train và suy giảm clarity/stability |
| 9.12 | Chọn checkpoint ứng viên | ML, QA | 0.5h | 9.10, 9.11 | Có bằng chứng validation + blind listening; không chỉ training loss |

### 10. Thử nghiệm 3 — điều chỉnh có kiểm soát (chỉ khi cần)

| ID | Công việc | Phụ trách | Ước lượng | Phụ thuộc | Đầu ra / tiêu chí nghiệm thu |
|---|---|---:|---:|---|---|
| 10.1 | Xác định một nguyên nhân thất bại chính | ML, QA | 1h | 9.10–9.12 | Chọn đúng một giả thuyết có bằng chứng |
| 10.2 | Chọn đúng một biến để thay đổi | ML | 0.25h | 10.1 | Chỉ một trong: LR 1e-4/2e-4, r 8/16, hoặc bỏ subset nhiễu |
| 10.3 | Giữ cố định các biến còn lại | ML | 0.25h | 10.2 | Config diff chứng minh chỉ một biến chính thay đổi |
| 10.4 | Chạy controlled experiment | ML | 0.5h + GPU | 10.3 | Run đầy đủ log và có thể tái lập |
| 10.5 | Đánh giá bằng cùng protocol | QA, RV | Theo bộ test | 10.4 | So sánh công bằng với ứng viên 9.12 |
| 10.6 | Chấp nhận hoặc bác bỏ thay đổi | ML, QA | 0.5h | 10.5 | Quyết định và bằng chứng được ghi vào run note |

### 11. Đánh giá cuối và cổng chất lượng

| ID | Công việc | Phụ trách | Ước lượng | Phụ thuộc | Đầu ra / tiêu chí nghiệm thu |
|---|---|---:|---:|---|---|
| 11.1 | Chốt adapter ứng viên cuối | ML | 0.25h | 9.12 hoặc 10.6 | Adapter path và config được đóng băng |
| 11.2 | Sinh lại toàn bộ test khóa kín | ML | 1h + GPU | 11.1, 1.10 | Dùng đúng prompt hash và inference config baseline |
| 11.3 | Tạo cặp base/fine-tune ẩn danh | QA | 0.5h | 1.7, 11.2 | Thứ tự A/B được randomize và lưu key riêng |
| 11.4 | Thu đánh giá từ ≥2 người nghe | RV | Theo bộ test | 11.3 | Mỗi câu có lựa chọn ưu tiên và score chi tiết |
| 11.5 | Tính tỷ lệ ưu tiên giọng | QA | 0.25h | 11.4 | Fine-tune được ưu tiên trên ≥70% câu |
| 11.6 | Kiểm tra transcript thủ công | QA | 1h | 11.2 | Không quá 1 lỗi từ quan trọng/100 từ |
| 11.7 | Tính tỷ lệ audio ổn định | QA | 0.5h | 11.2 | ≥95% không lặp, bỏ câu, silence dài hoặc noise collapse |
| 11.8 | Xác nhận khả năng tổng quát | ML, QA | 0.5h | 11.5–11.7 | Ngưỡng đạt trên test khóa kín, gồm category khó |
| 11.9 | Ghi quyết định pass/fail | P, ML, QA | 0.5h | 11.5–11.8 | Scorecard nêu rõ mọi gate; fail quay lại 10.1 hoặc data work |

### 12. Đóng gói, merge và phát hành

| ID | Công việc | Phụ trách | Ước lượng | Phụ thuộc | Đầu ra / tiêu chí nghiệm thu |
|---|---|---:|---:|---|---|
| 12.1 | Đóng gói adapter và config | ML | 0.5h | 11.9 pass | Thư mục run chứa adapter, config, revision và hướng dẫn load |
| 12.2 | Chọn reference audio sạch 3–10 giây | QA | 0.5h | 11.9 pass | Một speaker, không nhiễu; transcript chính xác tuyệt đối |
| 12.3 | Tạo `voices.json` | ML | 0.5h | 12.2 | Preset load được và trỏ đúng giọng mặc định |
| 12.4 | Chạy inference từ adapter đóng gói | ML | 0.5h | 12.1, 12.3 | Cài/load từ artifact và sinh audio thành công |
| 12.5 | Lưu bản rollback adapter đã duyệt | ML | 0.25h | 12.4 | Có bản bất biến kèm checksum/config |
| 12.6 | Merge LoRA vào base model (tùy chọn) | ML | 1h + GPU | 12.4 | `merge_lora.py` hoàn tất, model merged load được |
| 12.7 | Chạy test khóa kín trên model merged | ML | 1h + GPU | 12.6 | Đủ toàn bộ audio, cùng inference protocol |
| 12.8 | So sánh merged với adapter | QA | 1h | 12.7 | Đầu ra đủ tương đồng; không suy giảm gate 11.5–11.8 |
| 12.9 | Viết model card | P, ML | 2h | 11.9 pass | Nêu base model, dữ liệu tổng hợp, metric, giới hạn, license, mục đích cấm |
| 12.10 | Kiểm tra không lộ dữ liệu riêng tư | P, QA | 0.5h | 12.1, 12.9 | Artifact/Git không chứa raw audio, PII hoặc checkpoint ngoài phạm vi |
| 12.11 | Lập manifest phát hành và checksum | ML | 0.5h | 12.1–12.10 | Danh sách file/version/hash đầy đủ |
| 12.12 | Phê duyệt phát hành | P | 0.5h | 12.9–12.11 | Consent/license/quality/privacy đều đạt; có người phê duyệt và ngày |

**Gate M5:** Chỉ phát hành khi 11.9 là `pass` và 12.9–12.12 đều `DONE`.

## 4. Thứ tự triển khai khuyến nghị

```text
Quyền & mục tiêu (0)
  → Baseline khóa kín (1)
  → Môi trường (2) ───────────────┐
  → Thu và làm sạch dữ liệu (3)   │
  → Chia tập cố định (4)          │
  → Filter/encode (5)             │
  → Nâng cấp training (6) ←───────┘
  → Smoke test (7)
  → Pilot (8)
  → Full run (9)
  → Controlled experiment (10, nếu cần)
  → Đánh giá cuối (11)
  → Đóng gói/phát hành (12)
```

## 5. Critical path

`0.3 → 1.10 → 3.4 → 3.18 → 4.7 → 5.14 → 6.16 → 7.10 → 8.8 → 9.12 → 11.9 → 12.12`

Các task có thể làm song song:

- 1.x (test set/baseline) và 2.x (môi trường) sau khi chốt phạm vi.
- 6.4–6.6, 6.9–6.13 có thể triển khai song song với giai đoạn chuẩn bị dữ liệu.
- Model card có thể viết khung sớm, nhưng chỉ chốt số liệu sau 11.9.

## 6. Checklist hoàn thành dự án

- [ ] Có consent và license phù hợp.
- [ ] Base-model baseline và test set khóa kín đã lưu.
- [ ] Dataset đạt mục tiêu 2–4 giờ accepted audio (hoặc có ngoại lệ được phê duyệt).
- [ ] Split train/validation/test cố định, theo session, không rò rỉ.
- [ ] Filter và encode có seed, kiểm tra count, schema và codec range.
- [ ] Training có validation, seed, truncation report, resume và run metadata.
- [ ] Smoke test end-to-end đạt trước pilot/full run.
- [ ] Checkpoint được chọn bằng validation và nghe ẩn danh.
- [ ] Độ giống ≥70%, lỗi từ ≤1/100 từ, độ ổn định ≥95% trên test khóa kín.
- [ ] Adapter đóng gói load/inference được; merged model (nếu có) không suy giảm.
- [ ] Có `voices.json`, model card, checksum, release manifest và bản rollback.
- [ ] Không có audio riêng tư/checkpoint bị commit nhầm.
