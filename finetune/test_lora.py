from pathlib import Path
from vieneu import Vieneu

output = Path("finetune/eval")
output.mkdir(parents=True, exist_ok=True)

tts = Vieneu(
    mode="standard",
    backbone_repo="pnnbao-ump/VieNeu-TTS-0.3B",
    backbone_device="cuda",
    gguf_filename=None,
    codec_repo="neuphonic/neucodec",
    codec_device="cpu",
)

try:
    tts.load_lora_adapter(
        "finetune/output/rtx3050ti-lora/best_adapter"
    )

    audio = tts.infer(
        text="Hôm nay trời dịu mát. Bạn muốn đi dạo cùng tôi không?",
        ref_audio="finetune/dataset/raw_audio/reference.wav",
        ref_text="Lời thoại chính xác trong file reference.wav.",
    )
    tts.save(audio, output / "lora_test.wav")
finally:
    tts.close()