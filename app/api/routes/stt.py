from typing import Literal, Annotated
import time
from fastapi import APIRouter, UploadFile, File, Form, Request, HTTPException, status
import logging

from app.core.config import settings, Language, ResponseFormat
from app.core.transcriber import transcribe
from app.core.formatters import format_transcription
from app.core.models import load_model_instance

router = APIRouter(prefix="/audio", tags=["stt"])

HF_TOKEN = settings.HF_TOKEN

logger = logging.getLogger(__name__)

async def get_timestamp_granularities(request: Request) -> list[Literal["segment", "word"]]:
    TIMESTAMP_GRANULARITIES_COMBINATIONS = [
        [],
        ["segment"],
        ["word"],
        ["word", "segment"],
        ["segment", "word"],
    ]
    form = await request.form()
    if form.get("timestamp_granularities[]") is None:
        return ["segment"]
    timestamp_granularities = form.getlist("timestamp_granularities[]")
    assert timestamp_granularities in TIMESTAMP_GRANULARITIES_COMBINATIONS, (
        f"{timestamp_granularities} is not a valid value for `timestamp_granularities[]`."
    )
    return timestamp_granularities


def apply_defaults(settings, model, language=None, response_format=None):
    if model is None:
        model = settings.WHISPER.model
    if language is None:
        language = settings.WHISPER_LANGUAGE
    if response_format is None:
        response_format = settings.WHISPER_RESPONSE_FORMAT
    return model, language, response_format


@router.post("/transcriptions")
async def transcript(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form("large-v2"),
    language: Annotated[Language, Form()] = None,
    prompt: Annotated[str, Form()] = None,
    response_format: Annotated[ResponseFormat, Form()] = None,
    temperature: Annotated[float, Form()] = 0.0,
    timestamp_granularities: Annotated[
        list[Literal["segment", "word"]],
        Form(alias="timestamp_granularities[]"),
    ] = ["segment"],
    stream: Annotated[bool, Form()] = False,
    hotwords: Annotated[str, Form()] = None,
    suppress_numerals: Annotated[bool, Form()] = True,
    highlight_words: Annotated[bool, Form()] = False,
    align: Annotated[bool, Form()] = True,
    diarize: Annotated[bool, Form()] = False,
    chunk_size: Annotated[int, Form()] = 30,
):
    model, language, response_format = apply_defaults(settings, model, language, response_format)
    timestamp_granularities = await get_timestamp_granularities(request)
    request_id = request.state.request_id
    logger.debug(f"Request ID: {request_id} - Received transcription request")
    start_time = time.time()  # Start the timer
    
    if not align:
        if response_format in ('vtt', 'srt', 'aud', 'vtt_json'):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail="Subtitles format ('vtt', 'srt', 'aud', 'vtt_json') requires alignment to be enabled."
            )
        
        if diarize:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail="Diarization requires alignment to be enabled."
            )

    # Determine if word timestamps are required
    word_timestamps = "word" in timestamp_granularities

    # Build ASR options
    asr_options = {
        "suppress_numerals": suppress_numerals,
        "temperatures": temperature,
        "word_timestamps": word_timestamps,
        "initial_prompt": prompt,
        "hotwords": hotwords,
    }

    model_load_time = time.time()

    # Get model instance (reuse if cached)
    model_instance = await load_model_instance(model)

    logger.info(f"Loaded model {model} in {time.time() - model_load_time:.2f} seconds")

    try:
        transcription = await transcribe(
            audio_file=file,
            asr_options=asr_options,
            language=language,
            whisper_model=model_instance,
            align=align,
            diarize=diarize,
            chunk_size=chunk_size,
            request_id=request_id
        )
    except Exception as e:
        logger.exception(f"Request ID: {request_id} - Transcription failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="An unexpected error occurred while processing the transcription request."
        ) from e

    total_time = time.time() - start_time
    logger.info(f"Request ID: {request_id} - Transcription process took {total_time:.2f} seconds")

    response = format_transcription(transcription, response_format, highlight_words=highlight_words)

    return response