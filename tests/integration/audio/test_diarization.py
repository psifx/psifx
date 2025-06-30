import os
from pathlib import Path

import pytest

from tests.integration.conftest import run_command


@pytest.mark.skipif(not os.getenv("HF_TOKEN"), reason="HF_TOKEN not available")
@pytest.mark.integration
def test_audio_diarization(audio_path: Path, output_dir: Path, diarization_rttm: Path):
    """Test audio diarization."""

    pytest.importorskip("pyannote.audio", reason="Pyannote not installed")

    diarization_path = output_dir / "diarization.rttm"

    run_command(
        "psifx", "audio", "diarization", "pyannote", "inference",
        "--audio", audio_path,
        "--diarization", diarization_path
    )

    assert audio_path.exists(), "Audio extraction failed"
    assert diarization_path.exists(), "Audio diarization failed"

    def parse_rttm_file(path: Path):
        lines = path.read_text().splitlines()
        content = []
        for line in lines:
            parts = line.strip().split()
            assert len(parts) == 10, f"Abnormal rttm line: {parts}"
            content.append({
                "start": float(parts[3]),
                "duration": float(parts[4]),
                "speaker": parts[7]
            })
        return content

    parsed_lines = parse_rttm_file(diarization_path)
    expected_entries = parse_rttm_file(diarization_rttm)

    for expected in expected_entries:
        matched = any(
            abs(expected["start"] - line["start"]) < 1 and
            abs(expected["duration"] - line["duration"]) < 1 and
            expected["speaker"] == line["speaker"]
            for line in parsed_lines
        )
        assert matched, f"Expected segment not found: {expected}"
