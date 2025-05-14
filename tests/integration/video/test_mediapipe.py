import os
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest
import cv2

from psifx import command

@pytest.fixture
def run_mediapipe_inference(video_path, output_dir):
    """Run Mediapipe inference and return the path to the output poses file."""

    pytest.importorskip("mediapipe", reason="Mediapipe not installed")

    poses_path = os.path.join(output_dir, "e2e_poses.tar.xz")

    with patch("sys.argv", [
        "psifx", "video", "pose", "mediapipe", "inference",
        "--video", video_path,
        "--poses", poses_path,
        "--overwrite"
    ]):
        command.main()

    return poses_path

@pytest.mark.integration
def test_video_mediapipe_inference(video_path, run_mediapipe_inference):
    """Test Mediapipe pose inference produces valid tar with expected contents."""
    poses_path = run_mediapipe_inference

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "CV2 is unable to open the original video"
    original_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    assert os.path.exists(poses_path), "Pose detection failed"
    assert tarfile.is_tarfile(poses_path), "Poses file is not a valid tar file"

    with tarfile.open(poses_path, 'r:xz') as tar:
        files = tar.getnames()
        filenames = [Path(f).name for f in files]
        assert len(files) > 0, "Poses tar file is empty"
        assert "edges.json" in filenames, "Missing edges.json"
        assert len(files) == original_frame_count + 1, "Abnormal number of files in tar"


@pytest.mark.integration
def test_video_mediapipe_visualization(video_path, output_dir, run_mediapipe_inference):
    """Test Mediapipe pose visualization creates a video with expected frame count."""
    poses_path = run_mediapipe_inference
    pose_vis_path = os.path.join(output_dir, "e2e_pose_vis.mp4")

    with patch("sys.argv", [
        "psifx", "video", "pose", "mediapipe", "visualization",
        "--video", video_path,
        "--poses", poses_path,
        "--visualization", pose_vis_path,
        "--overwrite"
    ]):
        command.main()

    assert os.path.exists(pose_vis_path), "Pose visualization failed"
    assert os.path.getsize(pose_vis_path) > 0, "Pose visualization file is empty"

    cap = cv2.VideoCapture(video_path)
    original_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    cap = cv2.VideoCapture(pose_vis_path)
    visualization_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    assert visualization_frame_count == original_frame_count, "Visualization video has a different number of frames"