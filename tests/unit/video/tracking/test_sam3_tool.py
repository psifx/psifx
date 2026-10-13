"""Unit tests for the SAM3 tracking streaming helpers."""

from pathlib import Path
import sys
import types

import numpy as np
from PIL import Image
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
    tool.compute_dtype = None
    tool.video_storage_device = "cpu"
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


@pytest.mark.unit
def test_map_chunk_object_ids_respects_max_num_objects():
    tool = make_tool()
    mask_a = np.array([[1, 0], [0, 0]], dtype=bool)
    mask_b = np.array([[0, 1], [0, 0]], dtype=bool)
    mask_c = np.array([[0, 0], [1, 0]], dtype=bool)

    chunk_outputs = {
        0: {"object_ids": [1, 2, 3], "masks": [mask_a, mask_b, mask_c]},
    }

    mapping, next_global_id = tool._map_chunk_object_ids(
        chunk_outputs=chunk_outputs,
        prev_last_global_masks={},
        iou_threshold=0.3,
        next_global_id=0,
        max_num_objects=2,
    )

    assert mapping == {1: 0, 2: 1}
    assert next_global_id == 2


@pytest.mark.unit
def test_configure_model_max_num_objects_updates_model_and_config():
    tool = make_tool()
    tracker_config = types.SimpleNamespace(max_num_objects=10000)
    model_config = types.SimpleNamespace(max_num_objects=10000, tracker_config=tracker_config)
    tool.model = types.SimpleNamespace(config=model_config, max_num_objects=10000)

    tool._configure_model_max_num_objects(2)

    assert tool.model.max_num_objects == 2
    assert tool.model.config.max_num_objects == 2
    assert tool.model.config.tracker_config.max_num_objects == 2


@pytest.mark.unit
def test_segment_chunk_tracks_all_frames_by_default():
    class _DummyOut:
        def __init__(self, frame_idx):
            self.frame_idx = frame_idx

    class _DummyProcessor:
        def init_video_session(self, **kwargs):
            return object()

        def add_text_prompt(self, inference_session, text):
            return inference_session

        def postprocess_outputs(self, inference_session, out):
            return {
                "object_ids": [7],
                "masks": [np.array([[1, 0], [0, 0]], dtype=bool)],
            }

    class _DummyModel:
        def __init__(self):
            self.kwargs = None

        def propagate_in_video_iterator(self, inference_session, **kwargs):
            self.kwargs = kwargs
            yield _DummyOut(frame_idx=0)

    tool = make_tool()
    tool.processor = _DummyProcessor()
    tool.model = _DummyModel()

    chunk = [Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8))]
    outputs = tool._segment_chunk(chunk=chunk, text_prompt="people")

    assert tool.model.kwargs == {}
    assert outputs[0]["object_ids"] == [7]


class RecordingVideoWriter(DummyVideoWriter):
    """DummyVideoWriter that keeps every instance, keyed by output file name."""

    written = {}

    def __init__(self, path, input_dict=None, output_dict=None, overwrite=False):
        super().__init__(path, input_dict=input_dict, output_dict=output_dict, overwrite=overwrite)
        RecordingVideoWriter.written[self.path.name] = self


class DummyVideoReader:
    """Stand-in for VideoReader, only used for its frame rate."""

    frame_rate = "15/1"

    def __init__(self, path, *args, **kwargs):
        self.path = path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _write_video(path: Path, frames):
    import cv2

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (width, height))
    assert writer.isOpened(), "OpenCV cannot write mp4v videos in this environment"
    for frame in frames:
        writer.write(frame)
    writer.release()


@pytest.mark.unit
def test_masks_id_correction_merges_reappearing_id_and_renumbers(tmp_path, monkeypatch):
    """A red person tracked as id 0, lost, then tracked again as id 5 must end
    up in one output file; the blue person (id 3) present the whole time must
    not be merged with it. Output files are renumbered 0, 1."""
    n_frames, height, width = 60, 120, 160
    person_a = (slice(30, 90), slice(20, 50))    # red, left
    person_b = (slice(30, 90), slice(100, 130))  # blue, right
    a_spans = {0: range(0, 20), 5: range(30, 60)}

    video, masks = [], {0: [], 3: [], 5: []}
    for t in range(n_frames):
        frame = np.full((height, width, 3), 128, dtype=np.uint8)
        frame[person_b] = (255, 0, 0)
        for obj_id in masks:
            masks[obj_id].append(np.zeros((height, width, 3), dtype=np.uint8))
        masks[3][t][person_b] = 255
        for obj_id, span in a_spans.items():
            if t in span:
                frame[person_a] = (0, 0, 255)
                masks[obj_id][t][person_a] = 255
        video.append(frame)

    video_path = tmp_path / "video.mp4"
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    _write_video(video_path, video)
    for obj_id, frames in masks.items():
        _write_video(mask_dir / f"{obj_id}.mp4", frames)

    RecordingVideoWriter.written = {}
    monkeypatch.setattr(sam3_tool_module, "VideoWriter", RecordingVideoWriter)
    monkeypatch.setattr(sam3_tool_module, "VideoReader", DummyVideoReader)

    report = make_tool().masks_id_correction(
        video_path=str(video_path),
        mask_dir=str(mask_dir),
        out_mask_dir=str(tmp_path / "merged"),
        use_osnet=False,
    )

    assert report["original_id_count"] == 3
    assert report["merged_id_count"] == 2
    assert report["osnet_used"] is False
    assert report["groups"] == {"0": [0, 5], "3": [3]}
    assert report["output_id_remap"] == {"0": 0, "3": 1}

    assert sorted(RecordingVideoWriter.written) == ["0.mp4", "1.mp4"]
    merged_a = RecordingVideoWriter.written["0.mp4"].frames
    assert len(merged_a) == n_frames
    assert merged_a[10].any() and merged_a[45].any()  # both fragments of the red person
    assert not merged_a[25].any()                     # the gap in between stays empty


@pytest.mark.unit
def test_group_chains_with_temporal_veto_refuses_simultaneous_ids():
    """A short fragment (2) looks like both 0 and 1, which coexist: the less
    trusted of the two merges must be refused instead of bridging 0 and 1."""
    tool = make_tool()
    bounds = {0: (0, 100), 1: (0, 100), 2: (120, 140)}

    canonical, vetoed = tool._group_chains_with_temporal_veto(
        merges=[(0, 2, 0.9), (1, 2, 0.8)],
        all_ids=[0, 1, 2],
        bounds=bounds,
        allowed_overlap_pairs=set(),
        max_tolerated_overlap_frames=8,
    )

    assert canonical == {0: 0, 1: 1, 2: 0}
    assert [(v["from_id"], v["into_id"]) for v in vetoed] == [(1, 2)]
    assert sorted(vetoed[0]["conflict_ids"]) == [0, 1]
