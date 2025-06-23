from pathlib import Path

import pytest
import cv2

from tests.integration.conftest import run_command


@pytest.fixture
def run_samurai_inference(video_multi_path: Path, output_dir: Path) -> Path:
    """Run Samurai tracking inference and return the path to the output mask directory."""

    mask_dir = output_dir / "mask_dir"

    run_command(
        "psifx", "video", "tracking", "samurai", "inference",
        "--video", video_multi_path,
        "--mask_dir", mask_dir,
        "--max_objects", "2",
        "--overwrite"
    )

    return mask_dir


@pytest.mark.integration
def test_samurai_inference(video_multi_path: Path, run_samurai_inference: Path):
    """Test Samurai tracking inference produces valid .mp4 mask files with correct frame count."""

    cap_orig = cv2.VideoCapture(str(video_multi_path))
    assert cap_orig.isOpened(), "CV2 is unable to open the original video"
    original_frame_count = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_orig.release()

    mask_dir = run_samurai_inference
    assert mask_dir.is_dir(), f"Samurai inference mask dir {mask_dir} does not exist"

    masks = [mask for mask in mask_dir.iterdir() if mask.is_file()]
    assert len(masks) == 2, f"There should be 2 masks, got {len(masks)} instead"

    for mask in masks:
        assert mask.is_file() and mask.suffix == ".mp4", f"{mask} is not an .mp4 video"

        cap_out = cv2.VideoCapture(str(mask))
        assert cap_out.isOpened(), f"CV2 is unable to open the mask {mask} video"
        output_frame_count = int(cap_out.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_out.release()

        assert output_frame_count == original_frame_count, (
            f"Mask {mask} video has {output_frame_count} frames, "
            f"expected {original_frame_count} to match input video"
        )


@pytest.mark.integration
def test_tracking_visualization(video_multi_path: Path, output_dir: Path, mask_dir: Path):
    """Test Samurai mask visualization creates a video with expected frame count."""
    mask_vis_path = output_dir / "mask_vis.mp4"

    run_command(
        "psifx", "video", "tracking", "visualization",
        "--video", video_multi_path,
        "--masks", mask_dir,
        "--visualization", mask_vis_path,
        "--overwrite"
    )

    assert mask_vis_path.exists(), "Mask visualization failed"
    assert mask_vis_path.stat().st_size > 0, "Mask visualization file is empty"

    cap = cv2.VideoCapture(str(video_multi_path))
    original_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    cap = cv2.VideoCapture(str(mask_vis_path))
    visualization_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    assert visualization_frame_count == original_frame_count, "Visualization video has a different number of frames"
