import os
from pathlib import Path
import re
from unittest.mock import patch

import pytest

from psifx import command
from rapidfuzz import fuzz


def parse_vtt(filepath):
    """Parse a VTT file into a list of (start, end, text) tuples."""
    content = Path(filepath).read_text(encoding='utf-8')
    entries = []
    blocks = content.strip().split("\n\n")
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) >= 2 and "-->" in lines[0]:
            timing = lines[0]
            text = " ".join(lines[1:]).strip()
            match = re.match(r"(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})", timing)
            if match:
                start, end = match.groups()
                entries.append((start, end, text))
    return entries


@pytest.mark.integration
def test_audio_transcription_openai(audio_path, output_dir, video_text):
    """Test audio transcription with openai."""

    whisper = pytest.importorskip("whisper", reason="Whisper not installed")

    transcription_path = os.path.join(output_dir, "e2e_transcription.vtt")

    with patch("sys.argv",
               ["psifx", "audio", "transcription", "whisper", "openai", "inference", "--audio", audio_path,
                "--transcription",
                transcription_path], ):
        command.main()

    assert os.path.exists(transcription_path), "Audio transcription failed"

    actual_entries = parse_vtt(transcription_path)
    actual_text = " ".join(text for (start, end, text) in actual_entries)

    similarity = fuzz.ratio(actual_text, video_text)

    assert similarity > 90, (
        f"Text similarity too low for transcription:\n"
        f"Expected: {video_text}\nActual:   {actual_text}\n"
        f"Similarity: {similarity:.2f}"
    )


@pytest.mark.integration
def test_audio_transcription_huggingface(audio_path, output_dir, video_text):
    """Test audio transcription with huggingface."""

    transcription_path = os.path.join(output_dir, "e2e_transcription.vtt")

    with patch("sys.argv",
               ["psifx", "audio", "transcription", "whisper", "huggingface", "inference", "--audio", audio_path,
                "--transcription",
                transcription_path], ):
        command.main()

    assert os.path.exists(transcription_path), "Audio transcription failed"

    actual_entries = parse_vtt(transcription_path)
    actual_text = " ".join(text for (start, end, text) in actual_entries)

    similarity = fuzz.ratio(actual_text, video_text)

    assert similarity > 90, (
        f"Text similarity too low for transcription:\n"
        f"Expected: {video_text}\nActual:   {actual_text}\n"
        f"Similarity: {similarity:.2f}"
    )
