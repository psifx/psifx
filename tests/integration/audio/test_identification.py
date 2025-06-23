import os
from pathlib import Path
import wave
import json

import pytest

from tests.integration.conftest import run_command

@pytest.mark.skipif(not os.getenv("HF_TOKEN"), reason="HF_TOKEN not available")
@pytest.mark.integration
def test_audio_identification(audio_path: Path, output_dir: Path, diarization_rttm: Path):
    """Test audio identification."""

    pytest.importorskip("pyannote.audio", reason="Pyannote not installed")

    # TEST SPLIT
    left_audio_path = output_dir / "left_audio.wav"
    right_audio_path = output_dir / "right_audio.wav"

    run_command(
        "psifx", "audio", "manipulation", "split",
        "--stereo_audio", audio_path,
        "--left_audio", left_audio_path,
        "--right_audio", right_audio_path
    )

    def assert_mono(filepath: Path):
        with wave.open(str(filepath), 'rb') as wf:
            assert wf.getnchannels() == 1, f"{filepath} is not mono"

    for path in (left_audio_path, right_audio_path):
        assert path.exists(), f"{path} not created"
        assert_mono(path)

    # TEST CONVERSION
    run_command(
        "psifx", "audio", "manipulation", "conversion",
        "--audio", audio_path,
        "--mono_audio", audio_path,
        "--overwrite"
    )

    assert_mono(audio_path)

    # TEST IDENTIFICATION
    identification_path = output_dir / "identification.json"

    run_command(
        "psifx", "audio", "identification", "pyannote", "inference",
        "--mixed_audio", audio_path,
        "--diarization", diarization_rttm,
        "--mono_audios", right_audio_path, left_audio_path,
        "--identification", identification_path
    )

    assert identification_path.exists(), "Identification failed"

    with identification_path.open() as f:
        data = json.load(f)

    expected_mapping = {
        "SPEAKER_00": "right_audio.wav",
        "SPEAKER_01": "left_audio.wav"
    }

    assert "mapping" in data, "Missing 'mapping' key in Identification json"
    assert data["mapping"] == expected_mapping, f"Identification unexpected mapping: {data['mapping']}"
    assert "agreement" in data, "Missing 'agreement' key in Identification json"
