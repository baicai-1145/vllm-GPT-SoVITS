import math
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

_MAX_EMBEDDING_DIM = 8192


class OpenAICreateSpeechRequest(BaseModel):
    input: str
    model: str | None = None
    voice: str | None = Field(
        default=None,
        description="Speaker/voice to use. For Qwen3-TTS: vivian, ryan, aiden, etc.",
    )
    instructions: str | None = Field(
        default=None,
        description="Instructions for voice style/emotion (maps to 'instruct' for Qwen3-TTS)",
    )
    response_format: Literal["wav", "pcm", "flac", "mp3", "aac", "opus"] = "wav"
    speed: float | None = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
    )
    stream_format: Literal["sse", "audio"] | None = "audio"
    stream: bool = Field(
        default=False,
        description=(
            "If true, stream raw PCM audio chunks as they are decoded. "
            "Requires response_format='pcm'. Speed adjustment is not supported when streaming."
        ),
    )

    # Qwen3-TTS specific parameters
    task_type: Literal["CustomVoice", "VoiceDesign", "Base"] | None = Field(
        default=None,
        description="TTS task type: CustomVoice, VoiceDesign, or Base (voice clone)",
    )
    language: str | None = Field(
        default=None,
        description="Language code (e.g., 'Chinese', 'English', 'Auto')",
    )
    ref_audio: str | None = Field(
        default=None,
        description="Reference audio for voice cloning (Base task). URL, base64, or file URI.",
    )
    ref_text: str | None = Field(
        default=None,
        description="Transcript of reference audio for voice cloning (Base task)",
    )
    x_vector_only_mode: bool | None = Field(
        default=None,
        description="Use speaker embedding only without in-context learning (Base task)",
    )
    speaker_embedding: list[float] | None = Field(
        default=None,
        max_length=_MAX_EMBEDDING_DIM,
        description="Pre-computed speaker embedding vector (1024-dim for 0.6B, "
        "2048-dim for 1.7B). Skips speaker encoder extraction from ref_audio. "
        "Implies x_vector_only_mode=True. Mutually exclusive with ref_audio.",
    )
    max_new_tokens: int | None = Field(
        default=None,
        description="Maximum tokens to generate",
    )
    initial_codec_chunk_frames: int | None = Field(
        default=None,
        ge=0,
        description="Per-request initial chunk size override. If null, computed dynamically based on server load.",
    )
    text_lang: str | None = Field(default=None, description="GPT-SoVITS target text language, e.g. en/zh/ja/auto")
    prompt_lang: str | None = Field(default=None, description="GPT-SoVITS reference text language")
    text_split_method: str | None = Field(default=None, description="GPT-SoVITS text split method, e.g. cut4/cut5")
    batch_size: int | None = Field(default=None, ge=1, description="GPT-SoVITS inference batch size")
    batch_threshold: float | None = Field(default=None, ge=0.0, description="GPT-SoVITS bucket threshold")
    split_bucket: bool | None = Field(default=None, description="GPT-SoVITS bucketed batching switch")
    parallel_infer: bool | None = Field(default=None, description="GPT-SoVITS parallel infer switch")
    repetition_penalty: float | None = Field(default=None, ge=0.0, description="GPT-SoVITS T2S repetition penalty")
    sample_steps: int | None = Field(default=None, ge=1, description="GPT-SoVITS vocoder sample steps")
    fragment_interval: float | None = Field(default=None, ge=0.0, description="GPT-SoVITS fragment interval")
    seed: int | None = Field(default=None, description="GPT-SoVITS random seed")

    @field_validator("stream_format")
    @classmethod
    def validate_stream_format(cls, v: str) -> str:
        if v == "sse":
            raise ValueError("'sse' is not a supported stream_format yet. Please use 'audio'.")
        return v

    @field_validator("speaker_embedding")
    @classmethod
    def validate_speaker_embedding(cls, v: list[float] | None) -> list[float] | None:
        if v is not None and not all(math.isfinite(x) for x in v):
            raise ValueError("'speaker_embedding' values must be finite (no NaN or Inf)")
        return v

    @model_validator(mode="after")
    def validate_embedding_constraints(self) -> "OpenAICreateSpeechRequest":
        if self.speaker_embedding is not None:
            if self.ref_audio is not None:
                raise ValueError("'speaker_embedding' and 'ref_audio' are mutually exclusive")
        return self

    @model_validator(mode="after")
    def validate_streaming_constraints(self) -> "OpenAICreateSpeechRequest":
        if self.stream:
            if self.response_format not in ("pcm", "wav"):
                raise ValueError(
                    "Streaming (stream=true) requires response_format='pcm' or 'wav'. "
                    f"Got response_format='{self.response_format}'."
                )
            if self.speed is None:
                self.speed = 1.0
            elif self.speed != 1.0:
                raise ValueError(
                    "Speed adjustment is not supported when streaming (stream=true). Set speed=1.0 or omit it."
                )
        return self


class CreateAudio(BaseModel):
    audio_tensor: np.ndarray
    sample_rate: int = 24000
    response_format: str = "wav"
    speed: float = 1.0
    stream_format: Literal["sse", "audio"] | None = "audio"
    base64_encode: bool = True

    class Config:
        arbitrary_types_allowed = True


class AudioResponse(BaseModel):
    audio_data: bytes | str
    media_type: str


# --- Batch Speech Models ---


class SpeechBatchItem(BaseModel):
    """Per-item input for batch speech. Only `input` is required;
    all other fields override the batch-level defaults when set."""

    input: str
    voice: str | None = None
    instructions: str | None = None
    response_format: Literal["wav", "pcm", "flac", "mp3", "aac", "opus"] | None = None
    speed: float | None = Field(default=None, ge=0.25, le=4.0)
    task_type: Literal["CustomVoice", "VoiceDesign", "Base"] | None = None
    language: str | None = None
    ref_audio: str | None = None
    ref_text: str | None = None
    x_vector_only_mode: bool | None = None
    max_new_tokens: int | None = None
    initial_codec_chunk_frames: int | None = Field(default=None, ge=0)
    text_lang: str | None = None
    prompt_lang: str | None = None
    text_split_method: str | None = None
    batch_size: int | None = Field(default=None, ge=1)
    batch_threshold: float | None = Field(default=None, ge=0.0)
    split_bucket: bool | None = None
    parallel_infer: bool | None = None
    repetition_penalty: float | None = Field(default=None, ge=0.0)
    sample_steps: int | None = Field(default=None, ge=1)
    fragment_interval: float | None = Field(default=None, ge=0.0)
    seed: int | None = None


class BatchSpeechRequest(BaseModel):
    """Top-level request for batch speech generation.
    Fields here act as shared defaults; per-item overrides win."""

    model: str | None = None
    items: list[SpeechBatchItem] = Field(..., min_length=1)
    voice: str | None = None
    instructions: str | None = None
    response_format: Literal["wav", "pcm", "flac", "mp3", "aac", "opus"] = "wav"
    speed: float | None = Field(default=1.0, ge=0.25, le=4.0)
    task_type: Literal["CustomVoice", "VoiceDesign", "Base"] | None = None
    language: str | None = None
    ref_audio: str | None = None
    ref_text: str | None = None
    x_vector_only_mode: bool | None = None
    max_new_tokens: int | None = None
    initial_codec_chunk_frames: int | None = Field(default=None, ge=0)
    text_lang: str | None = None
    prompt_lang: str | None = None
    text_split_method: str | None = None
    batch_size: int | None = Field(default=None, ge=1)
    batch_threshold: float | None = Field(default=None, ge=0.0)
    split_bucket: bool | None = None
    parallel_infer: bool | None = None
    repetition_penalty: float | None = Field(default=None, ge=0.0)
    sample_steps: int | None = Field(default=None, ge=1)
    fragment_interval: float | None = Field(default=None, ge=0.0)
    seed: int | None = None


class SpeechBatchItemResult(BaseModel):
    index: int
    status: Literal["success", "error"]
    audio_data: str | None = None
    media_type: str | None = None
    error: str | None = None


class BatchSpeechResponse(BaseModel):
    id: str
    results: list[SpeechBatchItemResult]
    total: int
    succeeded: int
    failed: int


class StreamingSpeechSessionConfig(BaseModel):
    """Configuration sent as the first WebSocket message for streaming TTS."""

    model: str | None = None
    voice: str | None = None
    task_type: Literal["CustomVoice", "VoiceDesign", "Base"] | None = None
    language: str | None = None
    instructions: str | None = None
    response_format: Literal["wav", "pcm", "flac", "mp3", "aac", "opus"] = "wav"
    speed: float | None = Field(default=1.0, ge=0.25, le=4.0)
    max_new_tokens: int | None = Field(default=None, ge=1)
    initial_codec_chunk_frames: int | None = Field(
        default=None,
        ge=0,
        description="Initial chunk size for reduced TTFA. Overrides stage config for this session.",
    )
    ref_audio: str | None = None
    ref_text: str | None = None
    x_vector_only_mode: bool | None = None
    speaker_embedding: list[float] | None = Field(
        default=None,
        max_length=_MAX_EMBEDDING_DIM,
        description="Pre-computed speaker embedding vector. Mutually exclusive with ref_audio.",
    )
    stream_audio: bool = Field(
        default=False,
        description=(
            "If true, send raw PCM audio chunks progressively over WebSocket. "
            "Requires response_format='pcm'. Speed adjustment is not supported when streaming."
        ),
    )
    split_granularity: Literal["sentence", "clause"] = Field(
        default="sentence",
        description=(
            "Text splitting granularity: 'sentence' splits on .!?。！？, "
            "'clause' also splits on CJK commas ， and semicolons ；."
        ),
    )

    @model_validator(mode="after")
    def validate_streaming_constraints(self) -> "StreamingSpeechSessionConfig":
        if self.stream_audio:
            if self.response_format != "pcm":
                raise ValueError(
                    "WebSocket streaming audio (stream_audio=true) requires response_format='pcm'. "
                    f"Got response_format='{self.response_format}'."
                )
            if self.speed is None:
                self.speed = 1.0
            elif self.speed != 1.0:
                raise ValueError("Speed adjustment is not supported when stream_audio=true. Set speed=1.0 or omit it.")
        return self
