import os
import wsgiref
import whisperx
from fastapi import UploadFile
import logging
import time
import tempfile
import torch
import gc

TEMP_DIR = os.path.join(os.path.dirname(__file__), '..', 'temp')
os.makedirs(TEMP_DIR, exist_ok=True)

from app.core.config import settings, Language
from app.core.models import check_device, load_align_model_cached, load_diarize_model_cached, CustomWhisperModel

logger = logging.getLogger(__name__)

def cleanup_cache_only():
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


async def transcribe(
    audio_file: UploadFile,
    batch_size: int = settings.WHISPER_BATCH_SIZE,
    chunk_size: int = 30,
    asr_options: dict = {},
    language: Language = settings.WHISPER_LANGUAGE,
    whisper_model: CustomWhisperModel = settings.WHISPER.model,
    align: bool = False,
    diarize: bool = False,
    request_id: str = "",
    task: str = "transcribe",
) -> dict:
    start_time = time.time()  # Start timing
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{audio_file.filename}", dir=TEMP_DIR) as temp_file:
        temp_file.write(audio_file.file.read())
        file_path = temp_file.name

    logger.info(f"Request ID: {request_id} - Saving uploaded file took {time.time() - start_time:.2f} seconds")

    audio = None
    
    try:
        logger.info(f"Request ID: {request_id} - Transcribing {audio_file.filename} with model: {whisper_model.model_size_or_path} and options: {asr_options}")
        model_loading_start = time.time()
        model = whisperx.load_model(
            whisper_arch=whisper_model.model_size_or_path,
            device=whisper_model.device,
            compute_type=whisper_model.compute_type,
            language=language,
            asr_options=asr_options,
            vad_model=settings.WHISPER.vad_model,
            vad_method=settings.WHISPER.vad_method,
            vad_options=settings.WHISPER.vad_options,
            model=whisper_model,
            task=task,
        )
        logger.info(f"Request ID: {request_id} - Loading model took {time.time() - model_loading_start:.2f} seconds")

        audio_loading_start = time.time()
        audio = whisperx.load_audio(file_path)
        logger.info(f"Request ID: {request_id} - Loading audio took {time.time() - audio_loading_start:.2f} seconds")

        transcription_start = time.time()
        result = model.transcribe(
            audio=audio,
            batch_size=batch_size,
            chunk_size=chunk_size,
            num_workers=settings.WHISPER.num_workers,
            language=language,
            task=task,
        )
        logger.info(f"Request ID: {request_id} - Transcription took {time.time() - transcription_start:.2f} seconds")

        if align or diarize:
            alignment_model_start = time.time()
            logger.info(f"Request ID: {request_id} - Loading alignment model")
            model_a, metadata = await load_align_model_cached(
                language_code=result["language"],
            )
            logger.info(f"Request ID: {request_id} - Alignment model loaded")
            logger.info(f"Request ID: {request_id} - Loading alignment model took {time.time() - alignment_model_start:.2f} seconds")

            alignment_start = time.time()
            aligned_result = whisperx.align(
                transcript=result["segments"],
                model=model_a,
                align_model_metadata=metadata,
                audio=audio,
                device=check_device(),
                return_char_alignments=False
            )
            result["segments"] = aligned_result["segments"]
            logger.info(f"Request ID: {request_id} - Alignment took {time.time() - alignment_start:.2f} seconds")

        if diarize:
            diarization_model_start = time.time()

            logger.info(f"Request ID: {request_id} - Loading diarization model")

            diarize_model = await load_diarize_model_cached(model_name="pyannote/speaker-diarization-3.1")

            logger.info(f"Request ID: {request_id} - Diarization model loaded. Loading took {time.time() - diarization_model_start:.2f} seconds. Starting diarization")

            diarize_start = time.time()

            diarize_segments = diarize_model(audio)

            result["segments"] = whisperx.assign_word_speakers(diarize_segments, aligned_result).get("segments")

            logger.info(f"Request ID: {request_id} - Diarization took {time.time() - diarize_start:.2f} seconds")

        result["text"] = '\n'.join([segment["text"].strip() for segment in result["segments"] if segment["text"].strip()])

        # Add id to each segment
        for i, segment in enumerate(result["segments"]):
            segment["id"] = i
            if 'speaker' in segment:
                segment['speaker'] = int(segment['speaker'].split('_')[1])

        logger.info(f"Request ID: {request_id} - Transcription completed for {audio_file.filename}")
    except Exception as e:
        logger.error(f"Request ID: {request_id} - Transcription failed for {audio_file.filename} with error: {e}")
        raise
    finally:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            logger.error(f"Request ID: {request_id} - Could not remove temporary file: {file_path}")
        
        if settings.WHISPER_AUDIO_CLEANUP and audio is not None:
            del audio
            logger.debug(f"Request ID: {request_id} - Audio data cleaned up")
        
        if settings.WHISPER_CACHE_CLEANUP:
            cleanup_cache_only()
            logger.debug(f"Request ID: {request_id} - Cache cleanup completed")
    print("result", result)

    return result