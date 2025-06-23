import tarfile
from pathlib import Path

import pytest
import cv2

from tests.integration.conftest import run_command


@pytest.fixture
def run_mediapipe_single_inference(video_single_path: Path, mask_path: Path, output_dir: Path) -> Path:
    """Run Mediapipe inference and return the path to the output poses file."""

    pytest.importorskip("mediapipe", reason="Mediapipe not installed")

    poses_path = output_dir / "poses.tar.gz"

    run_command(
        "psifx", "video", "pose", "mediapipe", "single-inference",
        "--video", video_single_path,
        "--poses", poses_path,
        "--overwrite"
    )

    return poses_path

@pytest.fixture
def run_mediapipe_single_inference_mask(video_single_path: Path, mask_path: Path, output_dir: Path) -> Path:
    """Run Mediapipe inference and return the path to the output poses file."""

    pytest.importorskip("mediapipe", reason="Mediapipe not installed")

    poses_path = output_dir / "poses.tar.gz"

    run_command(
        "psifx", "video", "pose", "mediapipe", "single-inference",
        "--video", video_single_path,
        "--mask", mask_path,
        "--poses", poses_path,
        "--overwrite"
    )

    return poses_path


@pytest.fixture
def run_mediapipe_multi_inference(video_multi_path: Path, mask_dir: Path, output_dir: Path) -> Path:
    """Run Mediapipe inference and return the path to the output poses directory."""

    pytest.importorskip("mediapipe", reason="Mediapipe not installed")

    poses_dir = output_dir / "poses_dir"

    run_command(
        "psifx", "video", "pose", "mediapipe", "multi-inference",
        "--video", video_multi_path,
        "--poses_dir", poses_dir,
        "--masks", mask_dir,
        "--overwrite"
    )

    return poses_dir


def check_poses(poses_path: Path, video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), "CV2 is unable to open the original video"
    original_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    assert poses_path.exists(), "Pose detection failed"
    assert tarfile.is_tarfile(poses_path), "Poses file is not a valid tar file"

    with tarfile.open(poses_path, 'r:gz') as tar:
        files = tar.getnames()
        filenames = [Path(f).name for f in files]
        assert len(files) > 0, "Poses tar file is empty"
        assert "edges.json" in filenames, "Missing edges.json"
        assert len(files) == original_frame_count + 1, "Abnormal number of files in tar"


@pytest.mark.integration
def test_video_mediapipe_single_inference(video_single_path: Path, run_mediapipe_single_inference: Path):
    """Test Mediapipe pose single inference produces valid tar with expected contents."""
    check_poses(run_mediapipe_single_inference, video_single_path)

@pytest.mark.integration
def test_video_mediapipe_single_inference_mask(video_single_path: Path, run_mediapipe_single_inference_mask: Path):
    """Test Mediapipe pose single inference produces valid tar with expected contents."""
    check_poses(run_mediapipe_single_inference_mask, video_single_path)

@pytest.mark.integration
def test_video_mediapipe_multi_inference(video_multi_path: Path, run_mediapipe_multi_inference: Path):
    """Test Mediapipe pose multi inference produces valid tar with expected contents."""
    for poses_path in run_mediapipe_multi_inference.iterdir():
        check_poses(poses_path, video_multi_path)


@pytest.mark.integration
def test_video_mediapipe_visualization(poses_dir: Path, video_multi_path: Path, output_dir: Path):
    """Test Mediapipe pose visualization creates a video with expected frame count."""
    pose_vis_path = output_dir / "pose_vis.mp4"

    run_command(
        "psifx", "video", "pose", "mediapipe", "visualization",
        "--video", video_multi_path,
        "--poses", poses_dir,
        "--visualization", pose_vis_path,
        "--overwrite"
    )

    assert pose_vis_path.exists(), "Pose visualization failed"
    assert pose_vis_path.stat().st_size > 0, "Pose visualization file is empty"

    cap = cv2.VideoCapture(str(video_multi_path))
    original_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    cap = cv2.VideoCapture(str(pose_vis_path))
    visualization_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    assert visualization_frame_count == original_frame_count, "Visualization video has a different number of frames"
