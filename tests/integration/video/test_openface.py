import shutil
import tarfile
from pathlib import Path

import pytest
import cv2

from tests.integration.conftest import run_command


@pytest.mark.skipif(shutil.which("FeatureExtraction") is None, reason="FeatureExtraction executable not found in PATH")
@pytest.fixture
def run_openface_single_inference(video_single_path: Path, output_dir: Path):
    """Run OpenFace single inference and return the path to the features file."""
    faces_path = output_dir / "faces.tar.gz"

    run_command("psifx", "video", "face", "openface", "single-inference",
                "--video", video_single_path,
                "--features", faces_path,
                "--overwrite")
    return faces_path


@pytest.mark.skipif(shutil.which("FeatureExtraction") is None, reason="FeatureExtraction executable not found in PATH")
@pytest.fixture
def run_openface_single_inference_mask(video_single_path: Path, mask_path: Path, output_dir: Path):
    """Run OpenFace single inference and return the path to the features file."""
    faces_path = output_dir / "faces.tar.gz"

    run_command("psifx", "video", "face", "openface", "single-inference",
                "--video", video_single_path,
                "--mask", mask_path,
                "--features", faces_path,
                "--overwrite")
    return faces_path


@pytest.mark.skipif(shutil.which("FeatureExtraction") is None, reason="FeatureExtraction executable not found in PATH")
@pytest.fixture
def run_openface_multi_inference(video_multi_path: Path, mask_dir: Path, output_dir: Path):
    """Run OpenFace multi inference and return the path to the features directory."""
    features_dir = output_dir / "features_dir"

    run_command("psifx", "video", "face", "openface", "multi-inference",
                "--video", video_multi_path,
                "--features_dir", features_dir,
                "--masks", mask_dir,
                "--overwrite")
    return features_dir


def check_faces(faces_path: Path, video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), "CV2 is unable to open the original video"
    original_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    assert faces_path.exists(), "Face detection failed"
    assert tarfile.is_tarfile(faces_path), "Faces file is not a valid tar file"

    with tarfile.open(faces_path, 'r:gz') as tar:
        files = tar.getnames()
        filenames = [Path(f).name for f in files]

        assert len(filenames) > 0, "Faces tar file is empty"
        assert "edges.json" in filenames, "Missing edges.json"
        assert len(filenames) == original_frame_count + 1, "Abnormal number of files in tar"


@pytest.mark.integration
def test_openface_single_inference(run_openface_single_inference: Path, video_single_path: Path):
    """Test OpenFace feature extraction single inference output."""
    check_faces(run_openface_single_inference, video_single_path)


@pytest.mark.integration
def test_openface_single_inference_mask(run_openface_single_inference_mask: Path, video_single_path: Path):
    """Test OpenFace feature extraction single inference output."""
    check_faces(run_openface_single_inference_mask, video_single_path)


@pytest.mark.integration
def test_openface_multi_inference(run_openface_multi_inference: Path, video_multi_path: Path):
    """Test OpenFace feature extraction multi inference output."""
    for openface_path in run_openface_multi_inference.iterdir():
        check_faces(openface_path, video_multi_path)


@pytest.mark.integration
def test_openface_visualization(faces_dir: Path, video_multi_path: Path, output_dir: Path):
    """Test OpenFace face visualization."""

    face_vis_path: Path = output_dir / "face_vis.mp4"

    run_command("psifx", "video", "face", "openface", "visualization",
                "--video", video_multi_path,
                "--features", faces_dir,
                "--visualization", face_vis_path,
                "--overwrite")

    assert face_vis_path.exists(), "Face visualization failed"
    assert face_vis_path.stat().st_size > 0, "Face visualization file is empty"

    cap = cv2.VideoCapture(str(video_multi_path))
    assert cap.isOpened(), "CV2 is unable to open the original video"
    original_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    cap = cv2.VideoCapture(str(face_vis_path))
    assert cap.isOpened(), "CV2 is unable to open the visualization video"
    visualization_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    assert visualization_frame_count == original_frame_count, "Visualization video has a different number of frames"
