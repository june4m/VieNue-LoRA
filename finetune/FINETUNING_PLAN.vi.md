# Kế hoạch Fine-tune VieNeu-TTS 0.3B bằng LoRA

Tài liệu này chuyển các script hiện có trong repository thành một quy trình fine-tune thực tế, có thể đo lường và tái lập, dành cho một giọng nói tiếng Việt tùy chỉnh.

> [!IMPORTANT]
> Mã nguồn trong `finetune/` hiện nhắm đến model PyTorch cũ `pnnbao-ump/VieNeu-TTS-0.3B`. Quy trình này **không fine-tune model ONNX `VieNeu-TTS-v3-Turbo`** đang được dùng mặc định trong dự án. Hãy sử dụng model 0.3B cho kế hoạch này, trừ khi mã huấn luyện đã được cập nhật để hỗ trợ backbone và codec khác.

## 1. Mục tiêu và định nghĩa thành công

### Mục tiêu

Điều chỉnh VieNeu-TTS-0.3B theo một người nói tiếng Việt cụ thể, đồng thời giữ được khả năng phát âm rõ ràng, nhịp đọc ổn định và chất lượng chấp nhận được với những câu chưa từng xuất hiện trong dữ liệu huấn luyện.

### Tiêu chí thành công tối thiểu

Trước khi huấn luyện, cần ghi lại kết quả của base model trên tập đánh giá. Chỉ chấp nhận checkpoint fine-tune khi đạt toàn bộ các điều kiện sau:

| Hạng mục | Điều kiện chấp nhận |
|---|---|
| Độ giống giọng | Người nghe ẩn danh ưu tiên giọng fine-tune hơn giọng gốc ở ít nhất 70% câu kiểm thử |
| Độ rõ lời | Không quá 1 lỗi từ quan trọng trên mỗi 100 từ khi kiểm tra transcript thủ công |
| Độ ổn định | Ít nhất 95% audio kiểm thử không lặp câu, bỏ câu, im lặng dài hoặc biến thành tiếng nhiễu |
| Khả năng tổng quát | Chất lượng được giữ trên tập test khóa kín, không dùng trong huấn luyện hoặc chọn checkpoint |
| Quyền sử dụng | Người nói đã đồng ý rõ ràng với việc huấn luyện và phạm vi phát hành model |

Đây là ngưỡng khởi đầu cho dự án, không phải benchmark TTS chung. Hãy điều chỉnh chặt hơn sau khi có baseline đầu tiên.

## 2. Phạm vi khuyến nghị

- Chỉ dùng một người nói và một môi trường thu âm cho model đầu tiên.
- Mục tiêu là 2–4 giờ audio đã qua kiểm duyệt; 30–60 phút chỉ phù hợp để chạy thử pipeline.
- Mỗi đoạn audio dài từ 3 đến 15 giây.
- Không có nhạc nền, tiếng vang, người nói chồng nhau, vỡ tiếng hoặc artifact do khử nhiễu quá mạnh.
- Transcript tiếng Việt phải khớp chính xác lời nói, bao gồm cả dấu câu.
- Khuyến nghị sử dụng GPU CUDA có ít nhất 12 GB VRAM.

Không trộn nhiều người nói trong cùng một lần huấn luyện, trừ khi bạn chủ động thiết kế và kiểm thử cơ chế điều kiện hóa đa người nói.

## 3. Các đầu ra cần lưu

Sau khi hoàn tất dự án, cần giữ các artifact sau:

```text
finetune/
├── dataset/
│   ├── raw_audio/
│   ├── metadata.csv
│   ├── metadata_cleaned.csv
│   └── metadata_encoded.csv
├── output/
│   └── <run_name>/
├── eval/
│   ├── prompts.csv
│   ├── base/
│   ├── checkpoints/
│   └── scorecard.csv
└── FINETUNING_PLAN.vi.md
```

Hai thư mục `finetune/dataset/` và `finetune/output/` đã được Git bỏ qua. Tuy nhiên, vẫn cần kiểm tra để bảo đảm audio riêng tư hoặc checkpoint không bị commit nhầm vào vị trí khác.

## 4. Giai đoạn A — tạo baseline

Thực hiện giai đoạn này trước khi thay đổi trọng số model.

1. Tạo một tập đánh giá khóa kín gồm 30–50 câu.
2. Bao gồm câu kể, câu hỏi, câu cảm thán, ngày tháng viết bằng chữ, tên riêng, từ vay mượn, câu ngắn và câu dài.
3. Giữ ít nhất 20 câu hoàn toàn khác nội dung huấn luyện.
4. Sinh audio bằng base model chưa chỉnh sửa.
5. Lưu audio, prompt, seed nếu có, cấu hình inference và điểm của người đánh giá.

Cấu trúc đề xuất cho `finetune/eval/prompts.csv`:

```csv
id|category|text
001|statement|Hôm nay trời dịu mát và có gió nhẹ.
002|question|Bạn muốn gặp nhau vào chiều mai không?
003|name|Chị Minh Anh đang làm việc tại Đà Nẵng.
```

Tuyệt đối không đưa tập đánh giá khóa kín vào dữ liệu huấn luyện.

## 5. Giai đoạn B — chuẩn bị môi trường

Chạy các lệnh sau từ thư mục gốc của repository:

```powershell
uv sync --group gpu
uv pip install peft
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA GPU')"
```

Các script huấn luyện có import PEFT nhưng PEFT hiện chưa được khai báo trong `pyproject.toml`, vì vậy cần cài riêng. Nếu CUDA không khả dụng, hãy dừng tại đây; huấn luyện bằng CPU không phải lựa chọn thực tế cho kế hoạch này.

Ghi lại môi trường cho mỗi thử nghiệm:

```powershell
nvidia-smi
uv run python --version
uv pip freeze
```

Lưu kết quả vào ghi chú của lần chạy để có thể tái lập model tốt sau này.

## 6. Giai đoạn C — thu thập và tổ chức dữ liệu

Đặt các bản ghi âm vào:

```text
finetune/dataset/raw_audio/audio_0001.wav
finetune/dataset/raw_audio/audio_0002.wav
```

Tạo `finetune/dataset/metadata.csv` theo định dạng:

```text
audio_0001.wav|Xin chào, hôm nay bạn có khỏe không?
audio_0002.wav|Tôi đang thử một giọng đọc tiếng Việt mới.
```

### Quy tắc dữ liệu

- Mỗi file phải có tên duy nhất.
- Không dùng ký tự `|` trong transcript vì script sử dụng ký tự này để phân cách trường.
- Transcript phải khớp hoàn toàn với bản ghi. Không tự sửa lỗi nói trong transcript.
- Nên cắt tại ranh giới câu tự nhiên, không cắt giữa từ hoặc nhịp thở.
- Chuẩn hóa toàn bộ audio về WAV, mono, 16 kHz trước khi encode.
- Loại bỏ các đoạn trùng hoặc gần trùng để tránh model ghi nhớ quá mức một câu.
- Cân bằng âm vị, độ dài câu, dấu câu, thanh điệu và mục đích diễn đạt.
- Giữ khoảng cách micro và mức âm lượng nhất quán trong quá trình thu.

### Chia tập trước khi encode

Nếu có nhiều người nói, phải chia dữ liệu độc lập theo người nói. Với dự án một người nói, hãy chia theo buổi thu hoặc file nguồn để các đoạn liền nhau trong cùng một lần thu không bị rò rỉ sang tập khác:

- Training: 90%
- Validation: 5%
- Test: 5%

`train.py` hiện chưa sử dụng các tập này. Dù vậy, vẫn cần lưu manifest chia tập; Giai đoạn E mô tả thay đổi cần thiết cho huấn luyện.

### Điều kiện qua vòng kiểm tra dữ liệu

Nghe kiểm tra ngẫu nhiên ít nhất 10% số đoạn và toàn bộ các đoạn bị công cụ tự động cảnh báo. Chỉ tiếp tục khi:

- số file audio bị thiếu bằng 0;
- số file không đọc được bằng 0;
- tỷ lệ transcript không khớp trong mẫu kiểm tra dưới 1%;
- các đoạn vỡ tiếng, nhiễu, vang hoặc chồng giọng đã được loại thủ công;
- nội dung giữa train, validation và test không trùng nhau.

## 7. Giai đoạn D — lọc và encode dữ liệu

Chạy bộ lọc có sẵn:

```powershell
uv run python finetune/data_scripts/filter_data.py
```

Xem báo cáo loại dữ liệu và nghe thử một số mẫu ở từng nhóm bị loại. Bộ lọc hiện tại loại:

- audio ngắn hơn 3 giây hoặc dài hơn 15 giây;
- transcript chứa chữ số;
- từ viết tắt viết hoa;
- transcript không có dấu câu kết thúc.

Đây là bộ lọc khá nghiêm ngặt. Hãy chuyển số và chữ viết tắt thành đúng nội dung mà người nói đã đọc, thay vì chỉ xóa chúng.

Tiếp theo, encode audio thành token NeuCodec:

```powershell
uv run python finetune/data_scripts/encode_data.py
```

### Các hạn chế encode cần xử lý

`encode_data.py` hiện tại:

- xáo trộn metadata ngẫu nhiên nhưng không đặt seed;
- chỉ encode tối đa 2.000 đoạn;
- ghi đè `metadata_encoded.csv`;
- sử dụng `neuphonic/neucodec` và phải duy trì tương thích với backbone 0.3B đã chọn.

Trong lần chạy thử, có thể giữ giới hạn 2.000 đoạn. Trước lần huấn luyện chính thức, nên sửa script để nhận `--max_samples` và `--seed`, hoặc đặt `max_samples` đủ lớn để encode toàn bộ tập train và cố định seed. Ghi lại repository cùng phiên bản package codec.

Sau khi encode, xác minh:

- số dòng encode bằng số mẫu train dự kiến;
- mọi dòng đều đọc được theo cấu trúc `filename|text|JSON-array`;
- mảng code không rỗng và các giá trị nằm trong miền hợp lệ;
- không có file validation hoặc test trong dữ liệu train đã encode.

## 8. Giai đoạn E — giúp quá trình huấn luyện đo lường được

Cấu hình hiện tại phù hợp để làm baseline chạy thử:

```python
model = "pnnbao-ump/VieNeu-TTS-0.3B"
r = 16
lora_alpha = 32
lora_dropout = 0.05
learning_rate = 2e-4
batch_size = 2
gradient_accumulation_steps = 1
max_steps = 5000
max_sequence_length = 2048
```

Trước lần chạy chính thức, cần xử lý các hạn chế sau:

1. `train.py` dùng toàn bộ dữ liệu encode để train và tắt evaluation. Cần thêm cách tải train/validation cố định và truyền `eval_dataset` vào `Trainer`.
2. Thêm và ghi lại seed cố định cho Python, PyTorch, lựa chọn dữ liệu và `TrainingArguments`.
3. Bộ tạo sequence âm thầm cắt dữ liệu sau 2.048 token. Cần đếm các mẫu bị cắt rồi loại hoặc chia lại; chỉ tăng giới hạn khi đã kiểm thử VRAM.
4. `use_4bit` có trong config nhưng chưa được dùng khi tải model. Không được mặc định cho rằng QLoRA đang hoạt động.
5. `bf16=True` yêu cầu phần cứng hỗ trợ. Nếu không hỗ trợ, chỉ chuyển sang FP16 sau khi smoke test xác nhận loss ổn định.
6. Khi quá trình train bị gián đoạn, tiếp tục từ checkpoint thay vì bắt đầu một thử nghiệm mới không được ghi lại.

### Hiểu đúng số bước huấn luyện

Ước lượng số epoch bằng công thức:

```text
epochs ≈ max_steps × effective_batch_size ÷ training_examples
effective_batch_size = per_device_batch_size × gradient_accumulation_steps × GPU_count
```

Với 2.000 mẫu, một GPU, batch size 2 và không accumulation, 5.000 step tương đương khoảng 5 epoch. Hãy dùng số epoch và diễn biến validation để quyết định điểm dừng, không nên dùng con số 5.000 một cách máy móc.

## 9. Giai đoạn F — chạy thử nghiệm

### Thử nghiệm 0: smoke test pipeline

- Dùng 50–100 đoạn sạch.
- Train 20–50 step.
- Xác nhận loss hữu hạn, chỉ tham số LoRA được cập nhật, checkpoint được lưu và adapter có thể tải để inference.
- Kết quả này không dùng để đánh giá chất lượng giọng.

### Thử nghiệm 1: pilot

- Dùng 30–60 phút audio sạch nhất.
- Giữ LoRA `r=16`, alpha `32`, dropout `0.05`.
- Bắt đầu với learning rate `2e-4`.
- Lưu checkpoint đủ thường xuyên để so sánh giai đoạn đầu, giữa và cuối.
- Kiểm tra độ giống giọng, phát âm sai, lặp nội dung và overfitting.

### Thử nghiệm 2: toàn bộ dữ liệu

- Dùng toàn bộ dữ liệu train đã được chấp nhận.
- Tính số step từ số mẫu và đặt mục tiêu ban đầu khoảng 3–5 lượt đi qua dữ liệu.
- Chọn checkpoint dựa trên validation loss kết hợp nghe đánh giá; không tự động lấy checkpoint cuối.

### Thử nghiệm 3 tùy chọn: chỉ thay đổi một yếu tố

Nếu Thử nghiệm 2 chưa đạt điều kiện, chỉ thay đổi một biến chính:

- learning rate: so sánh `1e-4` với `2e-4`; hoặc
- dung lượng LoRA: so sánh `r=8` với `r=16`; hoặc
- dữ liệu: loại một nhóm dữ liệu nhiễu đã được ghi nhận.

Không thay đổi dữ liệu, rank, learning rate và số step cùng lúc vì sẽ không xác định được yếu tố nào tạo ra kết quả.

Bắt đầu huấn luyện bằng:

```powershell
uv run python finetune/train.py
```

Với mỗi lần chạy, ghi lại:

- tên lần chạy và thời gian;
- Git commit;
- revision của base model và tokenizer;
- phiên bản codec và các dependency;
- manifest hoặc hash của dataset và tổng thời lượng được chấp nhận;
- seed cùng mọi hyperparameter;
- loại GPU và VRAM cao nhất đã dùng;
- training loss, validation loss và checkpoint được chọn;
- bảng điểm đánh giá.

## 10. Giai đoạn G — đánh giá checkpoint

So sánh base model, ít nhất ba checkpoint và ứng viên cuối cùng bằng cùng prompt và cấu hình inference.

### Bảng điểm nghe thử

Nên có ít nhất hai người nghe chấm mỗi mẫu từ 1 đến 5:

| Tiêu chí | Nội dung đánh giá |
|---|---|
| Độ giống | Có giống người nói mục tiêu không? |
| Độ rõ | Từ ngữ và thanh điệu có dễ hiểu không? |
| Độ tự nhiên | Nhịp, trọng âm và cách ngắt câu có giống người thật không? |
| Độ ổn định | Có lặp, bỏ từ, nhiễu hoặc im lặng dài không? |
| Ngữ điệu | Câu hỏi, câu kể và cảm xúc có được thể hiện phù hợp không? |

Đổi tên file ngẫu nhiên để người đánh giá không biết checkpoint nào đã tạo audio. Chấm riêng từng nhóm khó vì điểm trung bình cao có thể che giấu lỗi với tên riêng hoặc câu dài.

### Kiểm tra overfitting

Loại checkpoint hoặc huấn luyện lại nếu:

- training loss giảm nhưng validation loss và chất lượng nghe xấu đi;
- model chèn câu trong dữ liệu train vào prompt không liên quan;
- chất lượng chỉ tốt với câu có nội dung gần giống dữ liệu train;
- checkpoint về sau giống giọng hơn nhưng độ rõ hoặc độ ổn định giảm.

## 11. Giai đoạn H — đóng gói và kiểm thử

Nên dùng trực tiếp adapter trước. Cách này giữ base model riêng biệt và giúp rollback dễ dàng.

```python
from vieneu import Vieneu

tts = Vieneu(backbone_repo="pnnbao-ump/VieNeu-TTS-0.3B")
tts.load_lora_adapter("finetune/output/<run_name>")
audio = tts.infer("Xin chào, đây là bài kiểm tra mô hình mới.")
tts.save(audio, "finetune/eval/final_test.wav")
```

Tạo voice preset có thể phân phối từ một đoạn tham chiếu sạch dài 3–10 giây với transcript chính xác:

```powershell
uv run python finetune/create_voices_json.py --audio reference.wav --text "Văn bản chính xác trong tệp tham chiếu." --name custom_voice --description "Vietnamese custom voice"
Copy-Item -LiteralPath .\voices.json -Destination .\finetune\output\<run_name>\voices.json
```

Chỉ merge model sau khi adapter đã vượt qua đánh giá:

```powershell
uv run python finetune/merge_lora.py --base_model pnnbao-ump/VieNeu-TTS-0.3B --adapter finetune/output/<run_name> --output finetune/output/<run_name>-merged
```

Chạy lại toàn bộ tập test khóa kín trên model đã merge. Quá trình merge chỉ hoàn tất khi đầu ra đủ tương đồng với adapter đã được chấp nhận.

## 12. Checklist phát hành

- [ ] Có văn bản đồng ý của người nói cho việc huấn luyện và phạm vi phát hành.
- [ ] Giấy phép dataset và model tương thích với mục đích sử dụng.
- [ ] Audio riêng tư và checkpoint không được đưa vào Git.
- [ ] Đã lưu baseline của base model.
- [ ] Tập train, validation và test được chia cố định, không rò rỉ.
- [ ] Dữ liệu vượt qua các điều kiện kiểm tra chất lượng.
- [ ] Đã ghi lại môi trường và revision model.
- [ ] Smoke test thành công trước khi chạy đầy đủ.
- [ ] Checkpoint được chọn bằng validation và nghe ẩn danh, không chỉ dựa vào training loss.
- [ ] Adapter và model đã merge đều vượt qua tập test khóa kín.
- [ ] `voices.json` có transcript chính xác và đúng giọng mặc định.
- [ ] Model card nêu rõ base model, tóm tắt dữ liệu, hạn chế, giấy phép và mục đích bị cấm.
- [ ] Giữ bản sao adapter cùng config đã được chấp nhận để rollback.

## 13. Lịch trình đề xuất

| Ngày | Công việc | Điều kiện hoàn thành |
|---|---|---|
| 1 | Xác định quyền sử dụng, mục tiêu, metric và tập prompt khóa kín | Có baseline và bảng điểm |
| 2–3 | Thu âm, cắt, chép lời và kiểm tra dữ liệu | Dữ liệu vượt qua kiểm tra |
| 4 | Lọc, encode, thêm cách chia tập và evaluation cố định | Số lượng và manifest khớp nhau |
| 5 | Smoke test và pilot | Adapter inference hoạt động từ đầu đến cuối |
| 6 | Train toàn bộ và đánh giá checkpoint | Có một ứng viên đạt chuẩn hoặc xác định được nguyên nhân thất bại |
| 7 | Thử nghiệm kiểm soát, đóng gói và viết model card | Checklist phát hành hoàn tất |

## 14. Checklist lệnh cho lần chạy đầu tiên


```powershell
# Chạy từ thư mục gốc repository
uv sync --group gpu
uv pip install peft

# Trước tiên, thêm WAV và tạo finetune/dataset/metadata.csv
uv run python finetune/data_scripts/filter_data.py
uv run python finetune/data_scripts/encode_data.py

# Sửa finetune/configs/lora_config.py với run_name riêng cho smoke test
uv run python finetune/train.py
```

Bước tiếp theo an toàn nhất là Thử nghiệm 0. Nó giúp xác nhận toàn bộ pipeline với chi phí thấp trước khi dành nhiều giờ encode dữ liệu hoặc chạy hàng nghìn GPU step trên một cấu hình có thể chưa tải, lưu hoặc inference đúng.
