from fastapi import APIRouter, UploadFile, File, Form
import tempfile
import os
import shutil

import whisperx
import gc
from whisperx.diarize import DiarizationPipeline

from app.core.config import settings

router = APIRouter(prefix="/stt", tags=["stt"])

HF_TOKEN = settings.HF_TOKEN

@router.post("/")
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    device: str = Form("cpu"),
    batch_size: int = Form(16),
    compute_type: str = Form("int8"),
    model_name: str = Form("large-v2")
):
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        shutil.copyfileobj(audio_file.file, temp_file)
        temp_audio_path = temp_file.name

    try:
        # 1. Transcribe with original whisper (batched)
        model = whisperx.load_model(model_name, device, compute_type=compute_type)

        # save model to local path (optional)
        # model_dir = "/path/"
        # model = whisperx.load_model(model_name, device, compute_type=compute_type, download_root=model_dir)

        audio = whisperx.load_audio(temp_audio_path)
        result = model.transcribe(audio, batch_size=batch_size)

        # delete model if low on GPU resources
        # import gc; import torch; gc.collect(); torch.cuda.empty_cache(); del model

        # 2. Align whisper output
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)


        # delete model if low on GPU resources
        # import gc; import torch; gc.collect(); torch.cuda.empty_cache(); del model_a

        # 3. Assign speaker labels
        diarize_model = DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)

        # add min/max number of speakers if known
        diarize_segments = diarize_model(audio)
        # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

        result = whisperx.assign_word_speakers(diarize_segments, result)
        print(diarize_segments)

        return result["segments"]
    finally:
        # Clean up temporary file
        os.unlink(temp_audio_path)