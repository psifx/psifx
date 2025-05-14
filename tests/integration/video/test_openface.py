import os
import shutil
import tarfile
from unittest.mock import patch
from pathlib import Path

import pytest
import cv2

from psifx import command

@pytest.mark.skipif(shutil.which("FeatureExtraction") is None, reason="FeatureExtraction executable not found in PATH")
@pytest.fixture
def run_openface_inference(video_path, output_dir):
    """Run OpenFace inference and return the path to the features file."""
    faces_path = os.path.join(output_dir, "e2e_faces.tar.xz")

    with patch("sys.argv", [
        "psifx", "video", "face", "openface", "inference",
        "--video", video_path,
        "--features", faces_path,
        "--overwrite"
    ]):
        command.main()

    return faces_path


@pytest.mark.integration
def test_openface_inference(run_openface_inference, video_path):
    """Test OpenFace feature extraction inference output."""

    faces_path = run_openface_inference

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "CV2 is unable to open the original video"
    original_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    assert os.path.exists(faces_path), "Face detection failed"
    assert tarfile.is_tarfile(faces_path), "Faces file is not a valid tar file"

    with tarfile.open(faces_path, 'r:xz') as tar:
        files = tar.getnames()
        filenames = [Path(f).name for f in files]

        assert len(filenames) > 0, "Faces tar file is empty"
        assert "edges.json" in filenames, "Missing edges.json"
        assert len(filenames) == original_frame_count + 1, "Abnormal number of files in tar"


@pytest.mark.integration
def test_openface_visualization(run_openface_inference, video_path, output_dir):
    """Test OpenFace face visualization."""

    faces_path = run_openface_inference
    face_vis_path = os.path.join(output_dir, "e2e_face_vis.mp4")

    with patch("sys.argv", [
        "psifx", "video", "face", "openface", "visualization",
        "--video", video_path,
        "--features", faces_path,
        "--visualization", face_vis_path,
        "--overwrite"
    ]):
        command.main()

    assert os.path.exists(face_vis_path), "Face visualization failed"
    assert os.path.getsize(face_vis_path) > 0, "Face visualization file is empty"

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "CV2 is unable to open the original video"
    original_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    cap = cv2.VideoCapture(face_vis_path)
    assert cap.isOpened(), "CV2 is unable to open the visualization video"
    visualization_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    assert visualization_frame_count == original_frame_count, "Visualization video has a different number of frames"