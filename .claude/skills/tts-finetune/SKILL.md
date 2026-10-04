---
name: tts-finetune
description: >
  Assist with fine-tuning TTS models in this repository.
  Use when working on TTS dataset preparation, preprocessing,
  training configuration, fine-tuning, checkpoint management,
  inference testing, evaluation, debugging training failures,
  or experiment documentation.
---

# TTS Fine-Tuning Skill

You are a TTS fine-tuning engineering assistant for this repository.

Your job is to help prepare, run, debug, evaluate, and document
TTS fine-tuning experiments while respecting the actual implementation
of this repository.

## Core Rule

Never assume training commands, configuration names, dataset formats,
model architecture, or checkpoint formats.

Before modifying or executing a training workflow:

1. Inspect the repository structure.
2. Read README and relevant documentation.
3. Locate training, preprocessing, inference, and evaluation scripts.
4. Inspect existing configuration files.
5. Determine the actual expected dataset format.
6. Only then propose or execute commands.

Prefer repository evidence over assumptions.

---

# Workflow

When asked to fine-tune the TTS model, follow these phases.

## Phase 1 — Repository Analysis

Identify:

- base model
- model architecture
- training framework
- Python version
- PyTorch version
- CUDA requirements
- dependency manager
- training entrypoint
- preprocessing scripts
- inference scripts
- configuration system
- checkpoint format
- existing fine-tuning support

Report findings before making major changes.

---

## Phase 2 — Environment Validation

Check:

- Python environment
- installed dependencies
- PyTorch
- CUDA availability
- GPU model
- GPU VRAM
- CUDA version
- ffmpeg if required
- required model files

When possible inspect GPU with:

```bash
nvidia-smi