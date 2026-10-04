# VieNeu-TTS 0.3B LoRA Fine-Tuning Plan

This guide turns the scripts in this repository into a practical, measurable fine-tuning project for a custom Vietnamese voice.

> [!IMPORTANT]
> The code under `finetune/` currently targets the legacy PyTorch model `pnnbao-ump/VieNeu-TTS-0.3B`. It does **not** fine-tune the default `VieNeu-TTS-v3-Turbo` ONNX model. Use the 0.3B model for this plan unless the training code is updated for another backbone and codec.

## 1. Goal and definition of success

### Goal

Adapt VieNeu-TTS-0.3B to one Vietnamese speaker while preserving clear pronunciation, stable rhythm, and acceptable output on text that was not present in training.

### Minimum success criteria

Before training, record the base-model results on the evaluation script. Accept a fine-tuned checkpoint only when it meets all of these gates:

| Area | Acceptance gate |
|---|---|
| Speaker similarity | Blind listeners prefer the fine-tuned voice to the base voice on at least 70% of test sentences |
| Intelligibility | No more than 1 material word error per 100 spoken words in a manual transcript check |
| Stability | At least 95% of test generations have no repeated phrase, skipped phrase, long silence, or collapse into noise |
| Generalization | Quality holds on a locked test set that was never used for training or checkpoint selection |
| Safety and rights | The speaker has explicitly consented to model training and intended distribution |

These are initial project gates, not universal TTS benchmarks. Tighten them after the first baseline.

## 2. Recommended project scope

- One speaker and one recording environment for the first model.
- 2–4 hours of accepted audio as the target; 30–60 minutes is enough only for a pipeline pilot.
- Clips between 3 and 15 seconds.
- Clean speech with no music, reverb, overlapping speakers, clipping, or aggressive denoising artifacts.
- Exact Vietnamese transcripts, including punctuation and the words actually spoken.
- A CUDA GPU with at least 12 GB VRAM is the repository's practical recommendation.

Do not combine speakers in one run unless multi-speaker conditioning is intentionally designed and tested.

## 3. Deliverables

At the end of the project, keep the following artifacts:

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
└── FINETUNING_PLAN.md
```

`finetune/dataset/` and `finetune/output/` are already ignored by Git. Confirm that no private recordings or model checkpoints are accidentally committed elsewhere.

## 4. Phase A — establish a baseline

Do this before changing weights.

1. Create a locked evaluation set of 30–50 sentences.
2. Cover statements, questions, exclamations, dates written as words, names, loanwords, short sentences, and long sentences.
3. Keep at least 20 sentences completely unrelated to the training script.
4. Generate them with the unmodified base model.
5. Save audio, prompt, random seed if available, inference settings, and reviewer scores.

Suggested `finetune/eval/prompts.csv` columns:

```csv
id|category|text
001|statement|Hôm nay trời dịu mát và có gió nhẹ.
002|question|Bạn muốn gặp nhau vào chiều mai không?
003|name|Chị Minh Anh đang làm việc tại Đà Nẵng.
```

Never train on this locked evaluation set.

## 5. Phase B — prepare the environment

Run commands from the repository root.

```powershell
uv sync --group gpu
uv pip install peft
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA GPU')"
```

The training scripts import PEFT, but PEFT is not currently declared in `pyproject.toml`, so install it explicitly. Stop here if CUDA is unavailable; CPU training is not a practical path for this plan.

Record the environment with every experiment:

```powershell
nvidia-smi
uv run python --version
uv pip freeze
```

Save the outputs in a run note. This makes a good result reproducible.

## 6. Phase C — collect and structure data

Place recordings here:

```text
finetune/dataset/raw_audio/audio_0001.wav
finetune/dataset/raw_audio/audio_0002.wav
```

Create `finetune/dataset/metadata.csv` in this format:

```text
audio_0001.wav|Xin chào, hôm nay bạn có khỏe không?
audio_0002.wav|Tôi đang thử một giọng đọc tiếng Việt mới.
```

### Data rules

- Use unique filenames.
- Do not put the `|` character in transcripts; the scripts use it as a delimiter.
- The transcript must match the recording exactly. Do not silently correct spoken mistakes.
- Prefer natural sentence boundaries instead of cutting a word or breath in half.
- Normalize all audio to mono, 16 kHz WAV before encoding.
- Avoid duplicate or near-duplicate clips; they overweight a phrase and can encourage memorization.
- Balance phonetic coverage, sentence length, punctuation, tone, and speaking intent.
- Keep recording gain and microphone distance consistent.

### Split before encoding

Create speaker-independent logic only if there are multiple speakers. For this single-speaker project, split by source recording/session so adjacent cuts from the same take do not leak across sets:

- Training: 90%
- Validation: 5%
- Test: 5%

The current `train.py` does not consume these splits yet. Preserve the split manifest anyway; Phase E describes the required training improvement.

### Data audit gate

Randomly inspect at least 10% of clips and all clips flagged by automatic checks. Do not proceed until:

- missing audio: 0;
- unreadable audio: 0;
- transcript mismatch in the inspected sample: below 1%;
- clipping, noise, echo, and overlap are rare enough to remove manually;
- train, validation, and test text do not overlap.

## 7. Phase D — filter and encode

Run the repository filter:

```powershell
uv run python finetune/data_scripts/filter_data.py
```

Review the rejection summary and inspect samples from every rejection category. The current filter rejects:

- clips outside 3–15 seconds;
- transcripts containing digits;
- uppercase acronyms;
- transcripts without final punctuation.

This is intentionally strict. Convert numbers and abbreviations into what the speaker actually says instead of merely deleting them.

Then encode audio into NeuCodec tokens:

```powershell
uv run python finetune/data_scripts/encode_data.py
```

### Encoding caveats that must be resolved

`encode_data.py` currently:

- randomly shuffles the metadata without a fixed seed;
- encodes at most 2,000 clips;
- overwrites `metadata_encoded.csv`;
- uses `neuphonic/neucodec` and must remain compatible with the selected 0.3B backbone.

For a quick pilot, keep the 2,000-clip cap. Before the production run, change the script to accept `--max_samples` and `--seed`, or set `max_samples` high enough for all accepted training clips and seed the shuffle. Record the codec repository and package version.

After encoding, verify:

- encoded row count equals the intended training row count;
- every row parses as `filename|text|JSON-array`;
- code arrays are non-empty and values are in the expected range;
- no validation or test clips were encoded into the training file.

## 8. Phase E — make training measurable

The current configuration is a useful smoke-test baseline:

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

Before the final run, address these limitations:

1. `train.py` uses all encoded rows for training and disables evaluation. Add deterministic train/validation loading and pass `eval_dataset` to `Trainer`.
2. Add and record a fixed seed for Python, PyTorch, dataset selection, and `TrainingArguments`.
3. The sequence builder silently truncates after 2,048 tokens. Count truncated examples and remove/re-segment them, or justify a longer sequence length after a VRAM test.
4. `use_4bit` exists in the config but is not used by model loading. Do not assume QLoRA is active.
5. `bf16=True` requires supported hardware. If unsupported, use FP16 only after a smoke test confirms stable loss.
6. Resume from a checkpoint for interrupted runs instead of starting a different, undocumented experiment.

### Understand the step count

Approximate epochs as:

```text
epochs ≈ max_steps × effective_batch_size ÷ training_examples
effective_batch_size = per_device_batch_size × gradient_accumulation_steps × GPU_count
```

With 2,000 examples, one GPU, batch size 2, and no accumulation, 5,000 steps is about 5 epochs. Use epochs and validation behavior—not the number 5,000 by itself—to decide when to stop.

## 9. Phase F — run experiments

### Experiment 0: pipeline smoke test

- Use 50–100 clean clips.
- Train for 20–50 steps.
- Confirm loss is finite, gradients update only LoRA parameters, a checkpoint saves, and the adapter can be loaded for inference.
- This experiment is not a quality result.

### Experiment 1: pilot

- Use 30–60 minutes of the cleanest audio.
- Keep LoRA `r=16`, alpha `32`, dropout `0.05`.
- Start at learning rate `2e-4`.
- Save frequently enough to compare early, middle, and late checkpoints.
- Listen for speaker similarity, pronunciation loss, repetition, and overfitting.

### Experiment 2: full data

- Use all accepted training data.
- Choose the step budget from the number of examples and target 3–5 passes through the data initially.
- Select the checkpoint using validation loss plus listening tests; do not automatically choose the last checkpoint.

### Optional Experiment 3: one controlled change

If Experiment 2 misses the gates, change only one major variable:

- learning rate: compare `1e-4` with `2e-4`; or
- LoRA capacity: compare `r=8` with `r=16`; or
- data: remove a documented noisy subset.

Do not change data, rank, learning rate, and step count at once—you will not know what caused the result.

Start training with:

```powershell
uv run python finetune/train.py
```

For each run, record:

- run name and timestamp;
- Git commit;
- base model and tokenizer revision;
- codec and dependency versions;
- dataset manifest/hash and accepted duration;
- seed and every hyperparameter;
- GPU model and peak VRAM;
- training/validation loss and checkpoint selected;
- evaluation scorecard.

## 10. Phase G — evaluate checkpoints

Compare the base model, at least three checkpoints, and the final candidate using identical prompts and inference settings.

### Listening scorecard

Have at least two listeners score each sample from 1–5:

| Dimension | What to judge |
|---|---|
| Similarity | Does it sound like the target speaker? |
| Intelligibility | Are words and tones understandable? |
| Naturalness | Does timing, stress, and phrasing sound human? |
| Stability | Are there repetitions, omissions, noise, or long silences? |
| Prosody | Are questions, statements, and emotion expressed appropriately? |

Randomize filenames so reviewers do not know which checkpoint produced a clip. Include difficult categories separately; a high average can hide failures on names or long sentences.

### Overfitting checks

Reject or retrain if:

- training loss improves while validation loss and listening quality worsen;
- the model copies training phrases into unrelated prompts;
- output quality is good only for sentences similar to training text;
- later checkpoints have stronger similarity but worse intelligibility or stability.

## 11. Phase H — package and test

Use the adapter directly first. This keeps the base model separate and makes rollback easy.

```python
from vieneu import Vieneu

tts = Vieneu(backbone_repo="pnnbao-ump/VieNeu-TTS-0.3B")
tts.load_lora_adapter("finetune/output/<run_name>")
audio = tts.infer("Xin chào, đây là bài kiểm tra mô hình mới.")
tts.save(audio, "finetune/eval/final_test.wav")
```

Create a portable voice preset from a clean 3–10 second reference whose transcript is exact:

```powershell
uv run python finetune/create_voices_json.py --audio reference.wav --text "Văn bản chính xác trong tệp tham chiếu." --name custom_voice --description "Vietnamese custom voice"
Copy-Item -LiteralPath .\voices.json -Destination .\finetune\output\<run_name>\voices.json
```

Only after the adapter passes evaluation, optionally merge it:

```powershell
uv run python finetune/merge_lora.py --base_model pnnbao-ump/VieNeu-TTS-0.3B --adapter finetune/output/<run_name> --output finetune/output/<run_name>-merged
```

Run the full locked test set again on the merged model. Merging is not complete until its outputs match the accepted adapter closely.

## 12. Release checklist

- [ ] Written speaker consent covers training and the intended release.
- [ ] Dataset license and model license are compatible with the use case.
- [ ] Private audio and checkpoints are excluded from Git.
- [ ] Base-model baseline is saved.
- [ ] Train/validation/test split is deterministic and leak-free.
- [ ] Dataset audit gates pass.
- [ ] Environment and model revisions are recorded.
- [ ] Smoke test passes before the full run.
- [ ] Checkpoint is selected by validation and blind listening, not training loss alone.
- [ ] Adapter and merged model both pass the locked test set.
- [ ] `voices.json` contains an exact transcript and intended default voice.
- [ ] Model card states base model, data summary, limitations, license, and prohibited uses.
- [ ] Rollback copy of the accepted adapter and configuration is retained.

## 13. Suggested schedule

| Day | Work | Exit condition |
|---|---|---|
| 1 | Define consent, use case, metrics, and locked prompts | Baseline and scorecard exist |
| 2–3 | Record, segment, transcribe, and audit data | Data gates pass |
| 4 | Filter, encode, add deterministic split/evaluation | Counts and manifests reconcile |
| 5 | Smoke test and pilot | End-to-end adapter inference works |
| 6 | Full run and checkpoint evaluation | One candidate passes or failure is diagnosed |
| 7 | Controlled follow-up, packaging, and model card | Release checklist passes |

## 14. First-run command checklist

```powershell
# From the repository root
uv sync --group gpu
uv pip install peft

# Add WAV files and create finetune/dataset/metadata.csv first
uv run python finetune/data_scripts/filter_data.py
uv run python finetune/data_scripts/encode_data.py

# Edit finetune/configs/lora_config.py for a uniquely named smoke test
uv run python finetune/train.py
```

The safest immediate next step is Experiment 0. It validates the complete pipeline cheaply before spending hours encoding data or thousands of GPU steps on a configuration that may not load, save, or infer correctly.
