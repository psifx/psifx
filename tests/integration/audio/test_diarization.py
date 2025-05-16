import os
from pathlib import Path
from unittest.mock import patch

import pytest

from psifx import command


@pytest.mark.integration
def test_audio_diarization(audio_path, output_dir, diarization_rttm):
    """Test audio diarization."""

    pyannote = pytest.importorskip("pyannote.audio", reason="Pyannote not installed")

    diarization_path = os.path.join(output_dir, "e2e_diarization.rttm")

    with patch("sys.argv",
               ["psifx", "audio", "diarization", "pyannote", "inference", "--audio", audio_path, "--diarization",
                diarization_path]):
        command.main()

    assert os.path.exists(audio_path), "Audio extraction failed"
    assert os.path.exists(diarization_path), "Audio diarization failed"

    def parse_rttm_file(path):
        lines = Path(path).read_text().splitlines()
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
