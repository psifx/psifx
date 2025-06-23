"""
Shared pytest fixtures for psifx integration tests.
"""

import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from psifx import command


def run_command(*args):
    with patch("sys.argv", list(map(str, args))):
        command.main()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for integration test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def video_single_path(temp_dir):
    """Return a temp copy of the test video file to avoid modifying the original."""
    original = Path(__file__).parent / "data" / "Single.mp4"
    temp_copy = temp_dir / "Single.mp4"
    shutil.copy(original, temp_copy)
    return temp_copy


@pytest.fixture
def video_multi_path(temp_dir):
    """Return a temp copy of the test video file to avoid modifying the original."""
    original = Path(__file__).parent / "data" / "Multi.mp4"
    temp_copy = temp_dir / "Multi.mp4"
    shutil.copy(original, temp_copy)
    return temp_copy


@pytest.fixture
def mask_dir(temp_dir):
    """Return a temp copy of the masks directory to avoid modifying the original."""
    original = Path(__file__).parent / "data" / "MaskDir"
    temp_copy = temp_dir / "MaskDir"
    shutil.copytree(original, temp_copy)
    return temp_copy


@pytest.fixture
def mask_path(temp_dir):
    """Return a temp copy of the mask video file to avoid modifying the original."""
    original = Path(__file__).parent / "data" / "1.mp4"
    temp_copy = temp_dir / "1.mp4"
    shutil.copy(original, temp_copy)
    return temp_copy


@pytest.fixture
def audio_path(temp_dir):
    """Return a temp copy of the test audio file to avoid modifying the original."""
    original = Path(__file__).parent / "data" / "Audio.wav"
    temp_copy = temp_dir / "Audio.wav"
    shutil.copy(original, temp_copy)
    return temp_copy


@pytest.fixture
def faces_dir(temp_dir):
    """Return a temp copy of the faces directory to avoid modifying the original."""
    original = Path(__file__).parent / "data" / "FacesDir"
    temp_copy = temp_dir / "FacesDir"
    shutil.copytree(original, temp_copy)
    return temp_copy


@pytest.fixture
def poses_dir(temp_dir):
    """Return a temp copy of the poses directory to avoid modifying the original."""
    original = Path(__file__).parent / "data" / "PosesDir"
    temp_copy = temp_dir / "PosesDir"
    shutil.copytree(original, temp_copy)
    return temp_copy


@pytest.fixture
def output_dir(temp_dir):
    """Create a directory for test outputs."""
    output_dir = temp_dir / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture
def diarization_rttm(temp_dir):
    """Create a temporary RTTM file with predefined speaker segments."""
    data = [
        {"start": 0.031, "duration": 11.121, "speaker": "SPEAKER_00"},
        {"start": 11.152, "duration": 12.032, "speaker": "SPEAKER_01"},
        {"start": 23.538, "duration": 11.441, "speaker": "SPEAKER_00"},
        {"start": 34.979, "duration": 10.918, "speaker": "SPEAKER_01"},
    ]

    rttm_path = temp_dir / "test.rttm"
    with rttm_path.open("w") as f:
        for entry in data:
            line = f"SPEAKER Audio 1 {entry['start']:.3f} {entry['duration']:.3f} <NA> <NA> {entry['speaker']} <NA> <NA>\n"
            f.write(line)

    return rttm_path


@pytest.fixture
def video_text():
    """Text in the video."""
    return ("If you can make one heap of all your winnings and risk it on one turn of pitch and toss and lose and start"
            " again at your beginnings and never breathe a word about your loss. If you can force your heart and nerve and sinew "
            "serve your turn long after they are gone and so hold on when there is nothing in you "
            "except a will which says to them hold on. If you can talk with crowds and keep your virtue or walk with kings nor lose the common touch. "
            "If neither foes nor loving friends can hurt you if all men count with you, but none too much. "
            "If you can fill the in forgiving minutes with 60 seconds worth of distance run your is the earth "
            "and everything that's in it and which is more you'll be a man my son.")
