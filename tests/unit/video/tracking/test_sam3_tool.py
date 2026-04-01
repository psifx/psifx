"""Unit tests for the SAM3 tracking streaming helpers."""

from pathlib import Path
import sys
import types

import numpy as np
import pytest

import transformers

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

if not hasattr(transformers, "Sam3VideoModel"):
    transformers.Sam3VideoModel = object
if not hasattr(transformers, "Sam3VideoProcessor"):
    transformers.Sam3VideoProcessor = object

import psifx.video.tracking.sam3.tool as sam3_tool_module
from psifx.video.tracking.sam3.tool import Sam3TrackingTool


class DummyVideoWriter:
    """In-memory stand-in for VideoWriter."""

    def __init__(self, path, input_dict=None, output_dict=None, overwrite=False):
        self.path = Path(path)
        self.input_dict = input_dict
        self.output_dict = output_dict
        self.overwrite = overwrite
        self.frames = []

    def write(self, image):
        self.frames.append(np.array(image, copy=True))

    def close(self):
        return None


def make_tool() -> Sam3TrackingTool:
    tool = Sam3TrackingTool.__new__(Sam3TrackingTool)
    tool.device = "cpu"
    tool.overwrite = True
    tool.verbose = False
    return tool


@pytest.mark.unit
def test_map_chunk_object_ids_reuses_previous_global_id():
    tool = make_tool()
    prev_mask = np.array([[1, 0], [0, 0]], dtype=bool)
    new_mask = np.array([[0, 1], [0, 0]], dtype=bool)

    chunk_outputs = {
        0: {"object_ids": [10], "masks": [prev_mask]},
        1: {"object_ids": [10, 11], "masks": [prev_mask, new_mask]},
    }
    prev_last_global_masks = {3: prev_mask}

    mapping, next_global_id = tool._map_chunk_object_ids(
        chunk_outputs=chunk_outputs,
        prev_last_global_masks=prev_last_global_masks,
        iou_threshold=0.3,
        next_global_id=4,
    )

    assert mapping[10] == 3
    assert mapping[11] == 4
    assert next_global_id == 5


@pytest.mark.unit
def test_write_chunk_masks_backfills_new_writer(monkeypatch):
    monkeypatch.setattr(sam3_tool_module, "VideoWriter", DummyVideoWriter)
    tool = make_tool()

    mask = np.array([[1, 0], [0, 0]], dtype=bool)
    chunk_outputs = {
        0: {"object_ids": [], "masks": []},
        1: {"object_ids": [7], "masks": [mask]},
    }
    id_mapping = {7: 2}
    writers = {}
    written_frames = {}

    tool._write_chunk_masks(
        chunk_outputs=chunk_outputs,
        id_mapping=id_mapping,
        writers=writers,
        written_frames=written_frames,
        mask_dir=Path("/tmp/masks"),
        frame_rate="25/1",
        frame_size=(2, 2),
        start_frame=2,
    )

    assert 2 in writers
    assert written_frames[2] == 4
    assert len(writers[2].frames) == 4

    # First 3 frames are backfill, last frame contains the detected mask.
    assert writers[2].frames[0].sum() == 0
    assert writers[2].frames[1].sum() == 0
    assert writers[2].frames[2].sum() == 0
    assert writers[2].frames[3].sum() > 0
