"""Exercise the optional OSNet path on real video, independently of SAM3 weights."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from psifx.io.video import VideoWriter
from psifx.video.tracking.sam3.tool import Sam3TrackingTool


@pytest.mark.integration
def test_osnet_relinks_real_person_without_merging_other_person(tmp_path):
    pytest.importorskip("torchreid")
    data = Path(__file__).parents[1] / "data"
    masks = tmp_path / "fragmented"
    masks.mkdir()

    # The fully visible person is deliberately lost for three frames and
    # reappears with a new ID. The other person's mask remains unchanged.
    for obj_id, source_id, first, last in [(0, 2, 0, 11), (5, 2, 14, 25), (3, 1, 0, 25)]:
        cap = cv2.VideoCapture(str(data / "MaskDir" / f"{source_id}.mp4"))
        assert cap.isOpened()
        writer = VideoWriter(
            masks / f"{obj_id}.mp4", input_dict={"-r": "25/1"},
            output_dict={"-c:v": "libx264", "-crf": "0", "-pix_fmt": "yuv420p"},
        )
        try:
            for frame_idx in range(25):
                ok, frame = cap.read()
                assert ok
                writer.write(frame if first <= frame_idx < last else np.zeros_like(frame))
        finally:
            cap.release()
            writer.close()

    # Correction is a post-processing operation and does not require loading SAM3.
    tool = Sam3TrackingTool.__new__(Sam3TrackingTool)
    tool.device, tool.overwrite, tool.verbose = "cpu", False, False
    corrected = tmp_path / "corrected"
    report = tool.masks_id_correction(
        video_path=str(data / "Multi.mp4"), mask_dir=str(masks),
        out_mask_dir=str(corrected), use_osnet=True,
    )
    assert report["osnet_used"] is True
    assert report["groups"] == {"0": [0, 5], "3": [3]}
    assert report["output_id_remap"] == {"0": 0, "3": 1}
    assert sorted(p.name for p in corrected.glob("*.mp4")) == ["0.mp4", "1.mp4"]

    for output_id in [0, 1]:
        cap = cv2.VideoCapture(str(corrected / f"{output_id}.mp4"))
        activity = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                activity.append(bool((frame[:, :, 0] > 127).any()))
        finally:
            cap.release()
        assert len(activity) == 25
        assert activity == ([True] * 11 + [False] * 3 + [True] * 11 if output_id == 0 else [True] * 25)
