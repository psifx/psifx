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
def test_map_chunk_object_ids_matches_object_first_seen_later_in_chunk():
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
    mask_stats = {}

    tool._write_chunk_masks(
        chunk_outputs=chunk_outputs,
        id_mapping=id_mapping,
        writers=writers,
        written_frames=written_frames,
        mask_stats=mask_stats,
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
    assert mask_stats[2]["non_empty_frames"] == 1
    assert mask_stats[2]["foreground_pixels"] == 1


@pytest.mark.unit
def test_extract_first_local_masks_handles_late_appearing_object():
    tool = make_tool()
    first_person_mask = np.array([[1, 0], [0, 0]], dtype=bool)
    second_person_mask = np.array([[0, 1], [0, 0]], dtype=bool)

    first_masks = tool._extract_first_local_masks(
        {
            0: {"object_ids": [10], "masks": [first_person_mask]},
            1: {"object_ids": [10, 11], "masks": [first_person_mask, second_person_mask]},
        }
    )

    assert list(first_masks.keys()) == [10, 11]
    assert np.array_equal(first_masks[10], first_person_mask)
    assert np.array_equal(first_masks[11], second_person_mask)


@pytest.mark.unit
def test_prune_mask_outputs_keeps_strongest_tracks(tmp_path):
    tool = make_tool()
    for idx in range(4):
        (tmp_path / f"{idx}.mp4").write_bytes(b"mask")

    kept_ids = tool._prune_mask_outputs(
        mask_dir=tmp_path,
        mask_stats={
            0: {"non_empty_frames": 5, "foreground_pixels": 20},
            1: {"non_empty_frames": 12, "foreground_pixels": 15},
            2: {"non_empty_frames": 12, "foreground_pixels": 40},
            3: {"non_empty_frames": 3, "foreground_pixels": 100},
        },
        max_num_objects=2,
    )

    assert kept_ids == [2, 1]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["1.mp4", "2.mp4"]
