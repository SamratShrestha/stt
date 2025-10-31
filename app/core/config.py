import secrets
import warnings
from typing import Annotated, Any, Literal

from pydantic import (
    AnyUrl,
    BeforeValidator,
    HttpUrl,
    computed_field,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Self


def parse_cors(v: Any) -> list[str] | str:
    if isinstance(v, str) and not v.startswith("["):
        return [i.strip() for i in v.split(",") if i.strip()]
    elif isinstance(v, list | str):
        return v
    raise ValueError(v)

from enum import Enum

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ResponseFormat(str, Enum):
    TEXT = "text"
    JSON = "json"
    VERBOSE_JSON = "verbose_json"
    VTT_JSON = "vtt_json"
    SRT = "srt"
    VTT = "vtt"
    AUD = "aud" # Audacity


class MediaType(str, Enum):
    APPLICATION_JSON = "application/json"
    TEXT_PLAIN = "text/plain"
    TEXT_VTT = "text/vtt"


class Language(str, Enum):
    AF = "af"
    AM = "am"
    AR = "ar"
    AS = "as"
    AZ = "az"
    BA = "ba"
    BE = "be"
    BG = "bg"
    BN = "bn"
    BO = "bo"
    BR = "br"
    BS = "bs"
    CA = "ca"
    CS = "cs"
    CY = "cy"
    DA = "da"
    DE = "de"
    EL = "el"
    EN = "en"
    ES = "es"
    ET = "et"
    EU = "eu"
    FA = "fa"
    FI = "fi"
    FO = "fo"
    FR = "fr"
    GL = "gl"
    GU = "gu"
    HA = "ha"
    HAW = "haw"
    HE = "he"
    HI = "hi"
    HR = "hr"
    HT = "ht"
    HU = "hu"
    HY = "hy"
    ID = "id"
    IS = "is"
    IT = "it"
    JA = "ja"
    JW = "jw"
    KA = "ka"
    KK = "kk"
    KM = "km"
    KN = "kn"
    KO = "ko"
    LA = "la"
    LB = "lb"
    LN = "ln"
    LO = "lo"
    LT = "lt"
    LV = "lv"
    MG = "mg"
    MI = "mi"
    MK = "mk"
    ML = "ml"
    MN = "mn"
    MR = "mr"
    MS = "ms"
    MT = "mt"
    MY = "my"
    NE = "ne"
    NL = "nl"
    NN = "nn"
    NO = "no"
    OC = "oc"
    PA = "pa"
    PL = "pl"
    PS = "ps"
    PT = "pt"
    RO = "ro"
    RU = "ru"
    SA = "sa"
    SD = "sd"
    SI = "si"
    SK = "sk"
    SL = "sl"
    SN = "sn"
    SO = "so"
    SQ = "sq"
    SR = "sr"
    SU = "su"
    SV = "sv"
    SW = "sw"
    TA = "ta"
    TE = "te"
    TG = "tg"
    TH = "th"
    TK = "tk"
    TL = "tl"
    TR = "tr"
    TT = "tt"
    UK = "uk"
    UR = "ur"
    UZ = "uz"
    VI = "vi"
    YI = "yi"
    YO = "yo"
    YUE = "yue"
    ZH = "zh"

# https://github.com/OpenNMT/CTranslate2/blob/master/docs/quantization.md
class Quantization(str, Enum):
    INT8 = "int8"
    INT8_FLOAT16 = "int8_float16"
    INT8_BFLOAT16 = "int8_bfloat16"
    INT8_FLOAT32 = "int8_float32"
    INT16 = "int16"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"
    DEFAULT = "default"


class Device(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


class VadMethod(str, Enum):
    SILERO = "silero"
    PYANNOTE = "pyannote"


class WhisperConfig(BaseModel):
    """See https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py#L599."""

    model: str = Field(default="large-v3")
    """
    Default Huggingface model to use for transcription. Note, the model must support being ran using CTranslate2.
    This model will be used if no model is specified in the request.

    Models created by authors of `faster-whisper` can be found at https://huggingface.co/Systran
    You can find other supported models at https://huggingface.co/models?p=2&sort=trending&search=ctranslate2 and https://huggingface.co/models?sort=trending&search=ct2
    """
    inference_device: Device = Field(default=Device.AUTO)
    device_index: int | list[int] = Field(default=0)
    compute_type: Quantization = Field(default=Quantization.DEFAULT)
    cpu_threads: int = Field(default=16)
    num_workers: int = Field(default=0)
    vad_method: VadMethod = Field(default=VadMethod.PYANNOTE)
    vad_model: str = Field(default=None)
    vad_options: dict = Field(default=None)
    cache: bool = Field(default=True)
    preload_model: str = Field(default=None)
    local_files_only: bool = Field(default=False)
    download_root: str = Field(default=None)


class AlignConfig(BaseModel):
    models: dict = Field(default_factory=dict)
    whitelist: list = Field(default_factory=list)
    cache: bool = Field(default=True)
    preload_model: str = Field(default=None)


class DiarizeConfig(BaseModel):
    cache: bool = Field(default=True)
    preload_model: str = Field(default=None)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ENVIRONMENT: Literal["local", "staging", "production"] = "local"

    BACKEND_CORS_ORIGINS: Annotated[
        list[AnyUrl] | str, BeforeValidator(parse_cors)
    ] = []


    @computed_field  # type: ignore[prop-decorator]
    @property
    def all_cors_origins(self) -> list[str]:
        origins = [str(origin).rstrip("/") for origin in self.BACKEND_CORS_ORIGINS]
        return origins

    PROJECT_NAME: str
    SENTRY_DSN: HttpUrl | None = None
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"

    HF_TOKEN: str

    # WhisperX settings
    WHISPER: WhisperConfig = WhisperConfig()
    WHISPER_LANGUAGE: Language | None = None
    WHISPER_MODEL: str = "large-v2"
    WHISPER_BATCH_SIZE: int = 16
    WHISPER_ALIGNMENT: AlignConfig = AlignConfig()
    WHISPER_DIARIZATION: DiarizeConfig = DiarizeConfig()
    WHISPER_CACHE_CLEANUP: bool = True
    WHISPER_AUDIO_CLEANUP: bool = True
    WHISPER_RESPONSE_FORMAT: ResponseFormat = ResponseFormat.JSON
    

    def _check_default_secret(self, var_name: str, value: str | None) -> None:
        if value == "changethis":
            message = (
                f'The value of {var_name} is "changethis", '
                "for security, please change it, at least for deployments."
            )
            if self.ENVIRONMENT == "local":
                warnings.warn(message, stacklevel=1)
            else:
                raise ValueError(message)

    @model_validator(mode="after")
    def _enforce_non_default_secrets(self) -> Self:
        self._check_default_secret("SECRET_KEY", self.SECRET_KEY)
        return self


settings = Settings()  # type: ignore