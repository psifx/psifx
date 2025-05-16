import os
from unittest.mock import patch

import pytest
import wave
import json

from psifx import command


@pytest.mark.integration
def test_audio_identification(audio_path, output_dir, diarization_rttm):
    """Test audio identification."""

    pyannote = pytest.importorskip("pyannote.audio", reason="Pyannote not installed")

    # TEST SPLIT

    left_audio_path = os.path.join(output_dir, "e2e_left_audio.wav")
    right_audio_path = os.path.join(output_dir, "e2e_right_audio.wav")

    with patch("sys.argv",
               ["psifx", "audio", "manipulation", "split", "--stereo_audio", audio_path, "--left_audio",
                left_audio_path,
                "--right_audio", right_audio_path]):
        command.main()

    def assert_mono(filepath):
        with wave.open(filepath, 'rb') as wf:
            assert wf.getnchannels() == 1, f"{filepath} is not mono"

    for path in (left_audio_path, right_audio_path):
        assert os.path.exists(path), f"{path} not created"
        assert_mono(path)

    # TEST CONVERSION
    with patch("sys.argv", ["psifx", "audio", "manipulation", "conversion", "--audio", audio_path, "--mono_audio",
                            audio_path,
                            "--overwrite"]):
        command.main()

    assert_mono(audio_path)

    # TEST IDENTIFICATION

    identification_path = os.path.join(output_dir, "e2e_identification.json")

    with patch("sys.argv",
               ["psifx", "audio", "identification", "pyannote", "inference", "--mixed_audio", audio_path,
                "--diarization", diarization_rttm, "--mono_audios", right_audio_path, left_audio_path,
                "--identification", identification_path]):
        command.main()

    assert os.path.exists(identification_path), "Identification failed"

    with open(identification_path) as f:
        data = json.load(f)

    expected_mapping = {
        "SPEAKER_00": "e2e_right_audio.wav",
        "SPEAKER_01": "e2e_left_audio.wav"
    }

    assert "mapping" in data, "Missing 'mapping' key in Identification json"
    assert data["mapping"] == expected_mapping, f"Identification unexpected mapping: {data['mapping']}"
    assert "agreement" in data, "Missing 'agreement' key in Identification json"
