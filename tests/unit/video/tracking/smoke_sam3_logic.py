"""Lightweight smoke test for SAM3 chunk-IoU stitching logic.

This script avoids loading SAM3, OpenCV, or pytest. It exercises only the
psifx logic that stitches chunk-local tracks into stable output ids and prunes
extra mask files when ``max_num_objects`` is used.
"""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import types

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def install_stubs() -> None:
    """Install minimal import-time stubs for optional runtime dependencies."""
    if "cv2" not in sys.modules:
        cv2 = types.ModuleType("cv2")
        cv2.COLOR_BGR2RGB = 0
        cv2.VideoCapture = object
        cv2.cvtColor = lambda frame, code: frame
        sys.modules["cv2"] = cv2

    if "skvideo.io" not in sys.modules:
        skvideo_module = types.ModuleType("skvideo")
        skvideo_io_module = types.ModuleType("skvideo.io")

        class _DummyFFmpegReader:
            INFO_AVERAGE_FRAMERATE = "avg_frame_rate"

            def __init__(self, *args, **kwargs):
                self.inputframenum = 0
                self.probeInfo = {"video": {self.INFO_AVERAGE_FRAMERATE: "25/1"}}

        class _DummyFFmpegWriter:
            def __init__(self, *args, **kwargs):
                return None

            def writeFrame(self, im):
                return None

        skvideo_io_module.FFmpegReader = _DummyFFmpegReader
        skvideo_io_module.FFmpegWriter = _DummyFFmpegWriter
        skvideo_module.io = skvideo_io_module
        sys.modules["skvideo"] = skvideo_module
        sys.modules["skvideo.io"] = skvideo_io_module

    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")
        torch.bfloat16 = "bfloat16"
        torch.float32 = "float32"
        torch.Tensor = type("Tensor", (), {})

        class _Serialization:
            @staticmethod
            def add_safe_globals(values):
                return None

        class _Cuda:
            @staticmethod
            def empty_cache():
                return None

            @staticmethod
            def ipc_collect():
                return None

        torch.serialization = _Serialization()
        torch.cuda = _Cuda()
        sys.modules["torch"] = torch

    if "transformers" not in sys.modules:
        transformers = types.ModuleType("transformers")
        transformers.Sam3VideoModel = object
        transformers.Sam3VideoProcessor = object
        sys.modules["transformers"] = transformers


def make_tool():
    from psifx.video.tracking.sam3.tool import Sam3TrackingTool

    tool = Sam3TrackingTool.__new__(Sam3TrackingTool)
    tool.device = "cpu"
    tool.overwrite = True
    tool.verbose = False
    return tool


def run_smoke_test() -> None:
    install_stubs()
    tool = make_tool()

    first_person_mask = np.array([[1, 0], [0, 0]], dtype=bool)
    second_person_mask = np.array([[0, 1], [0, 0]], dtype=bool)

    chunk_outputs = {
        0: {"object_ids": [10], "masks": [first_person_mask]},
        1: {"object_ids": [10, 11], "masks": [first_person_mask, second_person_mask]},
    }
    prev_last_global_masks = {3: first_person_mask, 4: second_person_mask}

    mapping, next_global_id = tool._map_chunk_object_ids(
        chunk_outputs=chunk_outputs,
        prev_last_global_masks=prev_last_global_masks,
        iou_threshold=0.3,
        next_global_id=5,
    )

    assert mapping[10] == 3
    assert mapping[11] == 4
    assert next_global_id == 5

    first_masks = tool._extract_first_local_masks(chunk_outputs)
    assert list(first_masks.keys()) == [10, 11]
    assert np.array_equal(first_masks[10], first_person_mask)
    assert np.array_equal(first_masks[11], second_person_mask)

    with TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        for idx in range(4):
            (tmp_dir / f"{idx}.mp4").write_bytes(b"mask")

        kept_ids = tool._prune_mask_outputs(
            mask_dir=tmp_dir,
            mask_stats={
                0: {"non_empty_frames": 5, "foreground_pixels": 20},
                1: {"non_empty_frames": 12, "foreground_pixels": 15},
                2: {"non_empty_frames": 12, "foreground_pixels": 40},
                3: {"non_empty_frames": 3, "foreground_pixels": 100},
            },
            max_num_objects=2,
        )

        assert kept_ids == [2, 1]
        assert sorted(path.name for path in tmp_dir.iterdir()) == ["1.mp4", "2.mp4"]

    print("sam3 lightweight logic smoke test passed")


if __name__ == "__main__":
    run_smoke_test()
