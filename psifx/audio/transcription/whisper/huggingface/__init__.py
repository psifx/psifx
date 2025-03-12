"""hugging face whisper module."""

from psifx.audio.transcription.whisper.huggingface.tool import HuggingFaceWhisperTool
from psifx.audio.transcription.whisper.huggingface.command import HuggingFaceWhisperCommand

__all__ = ["HuggingFaceWhisperTool", "HuggingFaceWhisperCommand"]