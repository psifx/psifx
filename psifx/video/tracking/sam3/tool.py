"""sam3 tracking tool."""

from collections import deque
from collections import OrderedDict, defaultdict
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import re
import cv2
import json

import torch
from PIL import Image
from transformers import Sam3VideoModel, Sam3VideoProcessor
from scipy.optimize import linear_sum_assignment

from psifx.io.video import VideoReader, VideoWriter
from psifx.utils.constants import SAM3_PATH
from psifx.video.tracking.tool import TrackingTool

# ----------------------------- Mask ID-correction ------------------------------------------
DEFAULT_MIN_FRAGMENT_FRAMES = 8
DEFAULT_MERGE_THRESHOLD = 0.6
DEFAULT_SIGNATURE_SAMPLES = 5
DEFAULT_POOLED_SAMPLES_PER_MEMBER = 5

# --- zeroth pass: same-time overlap (see `masks_id_correction`) ---
# minimum pixel-to-pixel distance between two masks, in px -- small
# means the two silhouettes actually touch/nearly touch.
DEFAULT_OVERLAP_PIXEL_GAP_THRESHOLD = 15.0
# distance between the two masks' centroids, in px -- what separates a
# same-body split from two different people occluding each other.
# Calibrated on labeled clinical sessions: true positives up to ~82px,
# closest negative at ~120px.
DEFAULT_OVERLAP_CENTROID_THRESHOLD = 115.0
# minimum number of frames where BOTH ids are simultaneously active for
# a pair's median distances to be trusted.
DEFAULT_MIN_OVERLAP_FRAMES = 8

# ----------------------------- Masks IO ----------------------------------------------------

# psifx writes plain "<int>.mp4" filenames; anything else in the
# directory is ignored rather than raising.
_MASK_FILENAME_RE = re.compile(r"^(\d+)\.mp4$")

# white-on-black decode threshold shared by every mask-reading function
# in this module.
DEFAULT_MASK_THRESHOLD = 127

# ----------------------------- Frame signature building ------------------------------------

# absolute cap on how far _search_nearby_signal looks on either side of
# an evenly-spaced target frame before giving up on that sampling slot.
MAX_POOLED_SEARCH_RADIUS = 200

# ----------------------------- Color / Hue Histogram Utilities -----------------------------

_HUE_HIST_BINS = 16  # 180deg/16 = 11.25deg/bin -- coarse enough to withstand
                      # lighting noise, fine enough to separate two distinct colors

# ----------------------------- OSNet embedding Utilities -----------------------------------

DEFAULT_MODEL_NAME = "osnet_x1_0"

# Below these dimensions (pixels) a crop is too small/too squashed to
# trust the resulting embedding -- same "no made-up signal" principle as
# `_mask_hue_histogram`: better None than a noisy embedding passed off
# as reliable.
MIN_CROP_W = 24
MIN_CROP_H = 48

# Minimum fraction of mask pixels inside the bbox for it to be worth
# zeroing out the background (below this threshold the polygon is
# probably too degenerate/noisy for reliable masking -- the raw
# bbox-crop is used anyway, the frame isn't discarded).
_MIN_MASK_FILL = 0.15

# ----------------------------- Masks merging -----------------------------------------------

# cost for a temporally-impossible pair (end doesn't precede start) in
# the Hungarian matrix -- worse than any real similarity score (cost =
# 1 - similarity, i.e. [0, 1]) but finite, so Hungarian never picks it
# ahead of a real candidate.
_IMPOSSIBLE_COST = 10.0



def _resolve_feature_extractor():
    """Resolves `FeatureExtractor`, handling two different layouts of
    the 'torchreid' package: the original project
    (github.com/KaiyangZhou/deep-person-reid) exposes `torchreid.utils`
    as a real subpackage, while the third-party PyPI "torchreid-pip"
    distribution (what plain `pip install torchreid` gets by default)
    hides everything under `torchreid.reid.*` and only rebinds `utils`
    as an attribute on the top-level package -- so
    `from torchreid.utils import FeatureExtractor` raises
    `ModuleNotFoundError` on that layout even though the package is
    installed correctly. Importing `torchreid` and accessing
    `torchreid.utils.FeatureExtractor` by attribute works on both."""
    import torchreid
    return torchreid.utils.FeatureExtractor

class OSNetEmbedder:
    """Minimal wrapper over `torchreid.utils.FeatureExtractor` for a
    single OSNet model.

        embedder = OSNetEmbedder(device="cpu")  # or "cuda"
        vec = embedder.embed(frame_bgr, bbox_xyxy, poly=poly)  # None if the crop is too poor to trust
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, model_path: str | None = None,
                 device: str = "cpu"):
        try:
            FeatureExtractor = _resolve_feature_extractor()
        except ImportError as exc:
            raise ImportError(
                "OSNet appearance embedding requires 'torch' and 'torchreid', "
                "not installed by default (heavy, optional dependency). "
                "Install with: pip install 'psifx[reid]' or "
                "pip install torch torchreid"
            ) from exc
        except AttributeError as exc:
            raise ImportError(
                "'torchreid' appears to be installed but does not expose "
                "'torchreid.utils.FeatureExtractor' -- likely that "
                "'pip install torchreid' installed the third-party PyPI "
                "distribution 'torchreid-pip' (a repackaging with a "
                "different layout, not the original deep-person-reid "
                "project) or an incomplete installation. Check with: "
                "python -c \"import torchreid; print(torchreid.utils.FeatureExtractor)\""
            ) from exc

        kwargs = dict(model_name=model_name, device=device, verbose=False)
        if model_path:
            kwargs["model_path"] = model_path
        # pretrained=True when model_path is absent: torchreid downloads
        # the weights from its model zoo on first use of a known
        # model_name (requires internet the first time, then cached
        # locally).
        self._extractor = FeatureExtractor(**kwargs)
        self.model_name = model_name
        self.device = device

    def _crop_person(self,frame_bgr: np.ndarray, bbox_xyxy: np.ndarray,
                    poly: np.ndarray | None) -> np.ndarray | None:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox_xyxy]
        x1, x2 = int(np.clip(x1, 0, w)), int(np.clip(x2, 0, w))
        y1, y2 = int(np.clip(y1, 0, h)), int(np.clip(y2, 0, h))
        if x2 - x1 < MIN_CROP_W or y2 - y1 < MIN_CROP_H:
            return None

        crop = frame_bgr[y1:y2, x1:x2].copy()
        if poly is not None and poly.shape[0] >= 3:
            shifted = poly.copy().astype(np.float64)
            shifted[:, 0] -= x1
            shifted[:, 1] -= y1
            mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
            cv2.fillPoly(mask, [np.round(shifted).astype(np.int32)], 255)
            fill = cv2.countNonZero(mask) / float(mask.size)
            if fill >= _MIN_MASK_FILL:
                crop[mask == 0] = 0  # zero out the background, the embedding focuses on the person

        return crop[:, :, ::-1]  # BGR (OpenCV) -> RGB (expected by torchreid)

    def embed(self, frame_bgr: np.ndarray, bbox_xyxy: np.ndarray,
               poly: np.ndarray | None = None) -> np.ndarray | None:
        """L2-normalized embedding (1D np.ndarray) of the person in
        `bbox_xyxy` (and optionally masked by `poly`) inside
        `frame_bgr`, or `None` if the crop is too small/degenerate to
        trust."""
        crop_rgb = self._crop_person(frame_bgr, bbox_xyxy, poly)
        if crop_rgb is None:
            return None
        features = self._extractor([crop_rgb])  # torch tensor (1, D), internal no_grad
        vec = features[0].detach().cpu().numpy().astype(np.float64)
        norm = np.linalg.norm(vec)
        if norm < 1e-9 or not np.isfinite(norm):
            return None
        return vec / norm

class Sam3TrackingTool(TrackingTool):
    def __init__(
        self,
        device: str = "cpu",
        model_path: str = SAM3_PATH,
        api_token: str = None,
        max_num_objects: Optional[int] = None,
        overwrite: bool = False,
        verbose: Union[bool, int] = True,
    ):
        super().__init__(
            device=device,
            overwrite=overwrite,
            verbose=verbose,
        )
        self.compute_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model_path = model_path
        self.api_token = api_token or os.environ.get("HF_TOKEN")
        self.max_num_objects = self._validate_positive_int(max_num_objects, "max_num_objects")
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        # Keep raw video frames on CPU to cap GPU memory usage for long clips.
        self.video_storage_device = "cpu" if self.device == "cuda" else self.device

        if self.verbose:
            print(f"Loading SAM3 model from '{self.model_path}' on device '{self.device}'")

        try:
            self._register_torch_safe_globals()
            self.model = Sam3VideoModel.from_pretrained(self.model_path, token=self.api_token).to(
                self.device, dtype=self.compute_dtype
            )
            self.processor = Sam3VideoProcessor.from_pretrained(self.model_path, token=self.api_token)
            if self.max_num_objects is not None:
                self._configure_model_max_num_objects(self.max_num_objects)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load SAM3 model. "
                "Check model access/token, or pass a local model path with --model_path."
            ) from exc

    def infer(
        self,
        video_path: Union[str, Path],
        mask_dir: Union[str, Path],
        text_prompt: str = "people",
        chunk_size: int = 300,
        iou_threshold: float = 0.3,
        id_correction: bool = True,
        merged_mask_dir: Optional[Union[str, Path]] = None,
        merge_report_path: Optional[Union[str, Path]] = None,
        id_correction_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Perform text-based segmentation and tracking from a video file.

        :param video_path: Path to the input video.
        :param mask_dir: Path to the output mask directory.
        :param text_prompt: Text description of objects to track.
        :param chunk_size: Number of frames to process at once.
        :param iou_threshold: IoU threshold for stitching chunks together.
        :param id_correction: Whether to merge the ids of a same object that
            tracking fragmented, into a second mask directory.
        :param merged_mask_dir: Output directory of the id-corrected masks,
            defaults to ``<mask_dir>_merged`` next to ``mask_dir``.
        :param merge_report_path: Path of the JSON report of the merge
            decisions, defaults to ``<mask_dir>_merge_report.json`` next to
            ``mask_dir``.
        :param id_correction_kwargs: Optional overrides of the
            ``masks_id_correction`` parameters (thresholds, ``use_osnet``, ...).
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}.")

        mask_dir = Path(mask_dir)
        if mask_dir.exists() and any(mask_dir.iterdir()):
            if self.overwrite:
                print(f"Mask directory {mask_dir} is non-empty")
            else:
                raise FileExistsError(f"Mask directory {mask_dir} is non-empty.")

        if id_correction:
            merged_mask_dir = (
                Path(merged_mask_dir) if merged_mask_dir is not None
                else mask_dir.parent / f"{mask_dir.name}_merged"
            )
            merge_report_path = (
                Path(merge_report_path) if merge_report_path is not None
                else mask_dir.parent / f"{mask_dir.name}_merge_report.json"
            )
            # Checked before tracking, not after: tracking can take hours.
            if not self.overwrite:
                if merged_mask_dir.exists() and any(merged_mask_dir.iterdir()):
                    raise FileExistsError(f"Merged mask directory {merged_mask_dir} is non-empty.")
                if merge_report_path.exists():
                    raise FileExistsError(f"Merge report {merge_report_path} already exists.")

        mask_dir.mkdir(parents=True, exist_ok=True)

        with VideoReader(path=video_path) as video_reader:
            frame_rate = video_reader.frame_rate

        writers: Dict[int, VideoWriter] = {}
        written_frames: Dict[int, int] = {}
        next_global_id = 0
        prev_last_global_masks: Dict[int, np.ndarray] = {}
        frame_size: Tuple[int, int] = (0, 0)
        processed_frame_count = 0

        try:
            for start_frame, chunk in self._iter_video_chunks(video_path, chunk_size):
                if not frame_size[0] and not frame_size[1]:
                    frame_size = chunk[0].size

                # If a chunk still OOMs, split and retry recursively in-order.
                pending_subchunks = deque([(start_frame, chunk)])

                while pending_subchunks:
                    sub_start_frame, sub_chunk = pending_subchunks.popleft()

                    try:
                        chunk_outputs = self._segment_chunk(sub_chunk, text_prompt)
                    except RuntimeError as exc:
                        if self._is_cuda_oom(exc) and self.device == "cuda" and len(sub_chunk) > 1:
                            self._clear_cuda_memory()
                            split_idx = len(sub_chunk) // 2
                            first_half = sub_chunk[:split_idx]
                            second_half = sub_chunk[split_idx:]
                            pending_subchunks.appendleft((sub_start_frame + split_idx, second_half))
                            pending_subchunks.appendleft((sub_start_frame, first_half))
                            if self.verbose:
                                print(
                                    "CUDA OOM while processing frames "
                                    f"{sub_start_frame}-{sub_start_frame + len(sub_chunk) - 1}; "
                                    f"retrying as chunks of {len(first_half)} and {len(second_half)} frames."
                                )
                            continue
                        raise

                    id_mapping, next_global_id = self._map_chunk_object_ids(
                        chunk_outputs=chunk_outputs,
                        prev_last_global_masks=prev_last_global_masks,
                        iou_threshold=iou_threshold,
                        next_global_id=next_global_id,
                        max_num_objects=self.max_num_objects,
                    )

                    self._write_chunk_masks(
                        chunk_outputs=chunk_outputs,
                        id_mapping=id_mapping,
                        writers=writers,
                        written_frames=written_frames,
                        mask_dir=mask_dir,
                        frame_rate=frame_rate,
                        frame_size=frame_size,
                        start_frame=sub_start_frame,
                    )

                    prev_last_global_masks = self._extract_last_global_masks(chunk_outputs, id_mapping)
                    processed_frame_count += len(sub_chunk)

                    del chunk_outputs
                    if self.device == "cuda":
                        self._clear_cuda_memory()
        finally:
            for writer in writers.values():
                writer.close()

        if processed_frame_count == 0:
            raise ValueError(f"No frames found in input video: {video_path}")
        if not writers:
            print("No masks to write.")
            return

        if id_correction:
            self._write_id_corrected_masks(
                video_path=video_path,
                mask_dir=mask_dir,
                merged_mask_dir=merged_mask_dir,
                merge_report_path=merge_report_path,
                **(id_correction_kwargs or {}),
            )

    def _write_id_corrected_masks(
        self,
        video_path: Union[str, Path],
        mask_dir: Path,
        merged_mask_dir: Path,
        merge_report_path: Path,
        **id_correction_kwargs,
    ):
        """
        Merge the ids tracking fragmented (see ``masks_id_correction``) into
        ``merged_mask_dir`` and write the report of the merge decisions. The
        raw masks in ``mask_dir`` are left untouched. When there is nothing
        usable to merge, the correction is skipped with a message instead of
        failing the whole tracking run.
        """
        # Built in a temporary sibling so an interrupted run never leaves a
        # half-written merged directory behind.
        tmp_dir = merged_mask_dir.parent / f".{merged_mask_dir.name}_tmp"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        try:
            report = self.masks_id_correction(
                video_path=str(video_path),
                mask_dir=mask_dir,
                out_mask_dir=tmp_dir,
                device=self.device,
                **id_correction_kwargs,
            )
        except ValueError as exc:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"ID correction skipped: {exc}")
            return
        report["output_mask_dir"] = str(merged_mask_dir)

        merged_mask_dir.mkdir(parents=True, exist_ok=True)
        for item in merged_mask_dir.iterdir():
            if _MASK_FILENAME_RE.match(item.name):
                item.unlink()
        for item in tmp_dir.iterdir():
            item.rename(merged_mask_dir / item.name)
        tmp_dir.rmdir()

        merge_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(merge_report_path, "w") as f:
            json.dump(report, f, indent=2)
        if self.verbose:
            print(
                f"ID correction: {report['original_id_count']} original ids -> "
                f"{report['merged_id_count']} after merging "
                f"({len(report['accepted_overlap_merges'])} overlap merge(s), "
                f"{len(report['accepted_merges'])} reappearance merge(s), "
                f"{len(report['vetoed_merges'])} vetoed). "
                f"Merged masks in {merged_mask_dir}, full report in {merge_report_path}"
            )

    @staticmethod
    def _iter_video_chunks(
        video_path: Union[str, Path], chunk_size: int
    ) -> Iterable[Tuple[int, List[Image.Image]]]:
        chunk: List[Image.Image] = []
        start_frame = 0
        with VideoReader(path=video_path) as video_reader:
            for frame in video_reader:
                chunk.append(Image.fromarray(frame))
                if len(chunk) >= chunk_size:
                    yield start_frame, chunk
                    start_frame += len(chunk)
                    chunk = []

            if chunk:
                yield start_frame, chunk

    def _segment_chunk(
        self,
        chunk: List[Image.Image],
        text_prompt: str,
    ):
        chunk_outputs = {idx: {"object_ids": [], "masks": []} for idx in range(len(chunk))}

        session = self.processor.init_video_session(
            video=chunk,
            inference_device=self.device,
            processing_device=self.device,
            video_storage_device=self.video_storage_device,
            dtype=self.compute_dtype,
        )
        try:
            self.processor.add_text_prompt(session, text_prompt)
            for out in self.model.propagate_in_video_iterator(session):
                processed = self.processor.postprocess_outputs(session, out)
                object_ids = self._to_int_list(processed["object_ids"])
                masks = self._to_bool_mask_list(processed["masks"])
                chunk_outputs[out.frame_idx] = {"object_ids": object_ids, "masks": masks}
        finally:
            del session

        return chunk_outputs

    @staticmethod
    def _to_int_list(ids) -> List[int]:
        if isinstance(ids, torch.Tensor):
            values = ids.detach().cpu().tolist()
        else:
            values = np.asarray(ids).tolist()
        return [int(value) for value in values]

    @staticmethod
    def _to_bool_mask_list(masks) -> List[np.ndarray]:
        if isinstance(masks, torch.Tensor):
            return [mask.detach().cpu().numpy().astype(bool) for mask in masks]
        return [np.asarray(mask).astype(bool) for mask in masks]

    @staticmethod
    def _compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute IoU between two binary masks."""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return float(intersection / union) if union > 0 else 0.0

    def _map_chunk_object_ids(
        self,
        chunk_outputs: Dict[int, Dict[str, List]],
        prev_last_global_masks: Dict[int, np.ndarray],
        iou_threshold: float,
        next_global_id: int,
        max_num_objects: Optional[int] = None,
    ) -> Tuple[Dict[int, int], int]:
        id_mapping: Dict[int, int] = {}

        curr_first_with_objects = None
        for frame_idx in sorted(chunk_outputs.keys()):
            frame_out = chunk_outputs[frame_idx]
            if frame_out["object_ids"]:
                curr_first_with_objects = frame_out
                break

        if curr_first_with_objects and prev_last_global_masks:
            used_global_ids = set()
            curr_ids = curr_first_with_objects["object_ids"]
            curr_masks = curr_first_with_objects["masks"]
            for curr_id, curr_mask in zip(curr_ids, curr_masks):
                best_iou = 0.0
                best_global_id = None
                for global_id, prev_mask in prev_last_global_masks.items():
                    if global_id in used_global_ids:
                        continue
                    iou = self._compute_mask_iou(prev_mask, curr_mask)
                    if iou > best_iou:
                        best_iou = iou
                        best_global_id = global_id

                if best_global_id is not None and best_iou >= iou_threshold:
                    id_mapping[curr_id] = best_global_id
                    used_global_ids.add(best_global_id)

        for frame_idx in sorted(chunk_outputs.keys()):
            frame_out = chunk_outputs[frame_idx]
            for obj_id in frame_out["object_ids"]:
                if obj_id not in id_mapping:
                    if max_num_objects is not None and next_global_id >= max_num_objects:
                        continue
                    id_mapping[obj_id] = next_global_id
                    next_global_id += 1

        return id_mapping, next_global_id

    def _configure_model_max_num_objects(self, max_num_objects: int) -> None:
        if hasattr(self.model, "config") and hasattr(self.model.config, "max_num_objects"):
            self.model.config.max_num_objects = max_num_objects

        tracker_config = getattr(getattr(self.model, "config", None), "tracker_config", None)
        if tracker_config is not None and hasattr(tracker_config, "max_num_objects"):
            tracker_config.max_num_objects = max_num_objects

        if hasattr(self.model, "max_num_objects"):
            self.model.max_num_objects = max_num_objects

        if self.verbose:
            print(f"Limiting SAM3 detections to at most {max_num_objects} object tracks.")

    @staticmethod
    def _validate_positive_int(value: Optional[int], name: str) -> Optional[int]:
        if value is None:
            return None
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}.")
        return value

    @staticmethod
    def _extract_last_global_masks(
        chunk_outputs: Dict[int, Dict[str, List]], id_mapping: Dict[int, int]
    ) -> Dict[int, np.ndarray]:
        for frame_idx in sorted(chunk_outputs.keys(), reverse=True):
            frame_out = chunk_outputs[frame_idx]
            if not frame_out["object_ids"]:
                continue

            global_masks = {}
            for local_id, local_mask in zip(frame_out["object_ids"], frame_out["masks"]):
                if local_id in id_mapping:
                    global_masks[id_mapping[local_id]] = local_mask
            return global_masks
        return {}

    def _write_chunk_masks(
        self,
        chunk_outputs: Dict[int, Dict[str, List]],
        id_mapping: Dict[int, int],
        writers: Dict[int, VideoWriter],
        written_frames: Dict[int, int],
        mask_dir: Path,
        frame_rate,
        frame_size: Tuple[int, int],
        start_frame: int,
    ):
        width, height = frame_size
        empty_mask_rgb = np.zeros((height, width, 3), dtype=np.uint8)

        for local_frame_idx in sorted(chunk_outputs.keys()):
            global_frame_idx = start_frame + local_frame_idx
            frame_out = chunk_outputs[local_frame_idx]

            masks_by_global_id: Dict[int, np.ndarray] = {}
            for local_obj_id, local_mask in zip(frame_out["object_ids"], frame_out["masks"]):
                global_obj_id = id_mapping.get(local_obj_id)
                if global_obj_id is not None:
                    masks_by_global_id[global_obj_id] = local_mask

            for global_obj_id in sorted(masks_by_global_id.keys()):
                if global_obj_id in writers:
                    continue
                writers[global_obj_id] = VideoWriter(
                    path=mask_dir / f"{global_obj_id}.mp4",
                    input_dict={"-r": frame_rate},
                    output_dict={"-c:v": "libx264", "-crf": "0", "-pix_fmt": "yuv420p"},
                    overwrite=self.overwrite,
                )
                written_frames[global_obj_id] = 0

                # Back-fill earlier frames so all mask videos keep identical frame counts.
                for _ in range(global_frame_idx):
                    writers[global_obj_id].write(image=empty_mask_rgb)
                    written_frames[global_obj_id] += 1

            for global_obj_id, writer in sorted(writers.items()):
                mask = masks_by_global_id.get(global_obj_id)
                if mask is None:
                    mask_rgb = empty_mask_rgb
                else:
                    mask_uint8 = (mask.astype(np.uint8) * 255)
                    mask_rgb = np.repeat(mask_uint8[..., np.newaxis], 3, axis=-1)
                writer.write(image=mask_rgb)
                written_frames[global_obj_id] += 1

    @staticmethod
    def _is_cuda_oom(exc: RuntimeError) -> bool:
        message = str(exc).lower()
        return "cuda" in message and "out of memory" in message

    @staticmethod
    def _clear_cuda_memory():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()

    @staticmethod
    def _register_torch_safe_globals() -> None:
        add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
        if add_safe_globals is None:
            return

        safe_globals = []
        try:
            from omegaconf.listconfig import ListConfig

            safe_globals.append(ListConfig)
        except Exception:
            pass

        try:
            from omegaconf.dictconfig import DictConfig

            safe_globals.append(DictConfig)
        except Exception:
            pass

        try:
            from omegaconf.base import ContainerMetadata

            safe_globals.append(ContainerMetadata)
        except Exception:
            pass

        try:
            import omegaconf.base as omegaconf_base

            safe_globals.extend(
                value for value in vars(omegaconf_base).values() if isinstance(value, type)
            )
        except Exception:
            pass

        try:
            import omegaconf.nodes as omegaconf_nodes

            safe_globals.extend(
                value
                for name, value in vars(omegaconf_nodes).items()
                if name.endswith("Node") and isinstance(value, type)
            )
        except Exception:
            pass

        safe_globals.extend([Any, list, dict, tuple, set, int, float, bool, str, bytes, OrderedDict, defaultdict])

        if safe_globals:
            add_safe_globals(safe_globals)

# -------------------------------------------------------------------------------------------
# ----------------------------- Mask ID-correction ------------------------------------------
# -------------------------------------------------------------------------------------------

    def masks_id_correction(
        self,
        *,
        video_path: str,
        mask_dir: str,
        out_mask_dir: str,
        min_fragment_frames: int = DEFAULT_MIN_FRAGMENT_FRAMES,
        merge_threshold: float = DEFAULT_MERGE_THRESHOLD,
        signature_samples: int = DEFAULT_SIGNATURE_SAMPLES,
        pooled_samples_per_member: int = DEFAULT_POOLED_SAMPLES_PER_MEMBER,
        overlap_pixel_gap_threshold: float = DEFAULT_OVERLAP_PIXEL_GAP_THRESHOLD,
        overlap_centroid_threshold: float = DEFAULT_OVERLAP_CENTROID_THRESHOLD,
        min_overlap_frames: int = DEFAULT_MIN_OVERLAP_FRAMES,
        overlap_classifier_path: str | None = None,
        device: str = "cpu",
        use_osnet: bool = True,
    ) -> dict:
        """Reads `mask_dir`, merges fragmented identities via appearance
        (OSNet embedding + hue histogram), writes a NEW MaskDir at
        `out_mask_dir` (never touches the original), and returns a
        JSON-able report of every merge decision made -- accepted and,
        for transparency, every candidate pair actually considered.

        Steps:
          0. Zeroth pass (`_resolve_overlap_merges`): ids alive at the
             same time that are really one body split in two parts.
          1. Pass 1 (`_resolve_merges`): a track that ENDS is matched
             against tracks that START later, by appearance, with one
             global Hungarian assignment.
          2. Pass 2 (`_resolve_group_merges`): start tracks pass 1 left
             unmatched are compared against whole pass-1 groups.
        Every union goes through `_group_chains_with_temporal_veto`,
        which refuses a merge that would put two simultaneously-active
        ids in one group (only zeroth-pass pairs may coexist).

        Mask videos are streamed from disk, never loaded whole in RAM."""
        mask_paths = self._list_mask_files(mask_dir)
        if not mask_paths:
            raise ValueError(f"No <id>.mp4 mask files found in {mask_dir}")
    
        all_ids = sorted(mask_paths.keys())
        bounds: dict[int, tuple[int, int]] = {}
        excluded_short: list[int] = []
        total_frames: int | None = None
        total_frames_source_id: int | None = None
        height = width = None
    
        for obj_id in all_ids:
            scan = self._scan_track(mask_paths[obj_id])
            if scan is None:
                continue
            first, last, real_frames, decoded_frames, h, w = scan
            
            if total_frames is None:
                total_frames = decoded_frames
                total_frames_source_id = obj_id
                height, width = h, w
    
            elif decoded_frames != total_frames:
                raise ValueError(
                    f"id {obj_id} has {decoded_frames} frames but id {total_frames_source_id} "
                    f"has {total_frames} -- every <id>.mp4 in a MaskDir is expected to be padded "
                    f"to the same total frame count."
                )
            if real_frames < min_fragment_frames:
                excluded_short.append(obj_id)
                continue
            bounds[obj_id] = (first, last)
    
        if not bounds:
            raise ValueError(
                f"No id in {mask_dir} has at least {min_fragment_frames} non-empty "
                f"frames -- nothing to merge (every id was too short/noisy)."
            )
    
        # --- zeroth pass: same-body fragments that coexist in time.
        # Computed before pass 1/2 so its unions are already folded into
        # `canonical` when pass 2 builds its candidate groups. An optional
        # classifier weights JSON (`overlap_classifier_path`) REPLACES the
        # two fixed thresholds (see `_resolve_overlap_merges`). ---

        classifier = None
        if overlap_classifier_path:
            with open(overlap_classifier_path) as f:
                classifier = json.load(f)
        overlap_accepted, overlap_rejected = self._resolve_overlap_merges(
            mask_paths, bounds, overlap_pixel_gap_threshold, overlap_centroid_threshold, min_overlap_frames,
            classifier=classifier,
        )
        overlap_merge_tuples = [(r["id_a"], r["id_b"], 0.0) for r in overlap_accepted]

        embedder = None
        if use_osnet:
            try:
                embedder = OSNetEmbedder(device=device)
            except ImportError as exc:
                print(f"WARNING [sam3 id correction] OSNet unavailable ({exc}) -- "
                        f"falling back to color-only matching.")
    
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")
    
        end_ids = list(bounds.keys())
        # a track starting at frame 0 is the video's first sighting of that
        # id, not a "reappearance" -- nothing plausible for it to resume, so
        # it's excluded from the START side (it can still be an END, i.e.
        # something else can resume INTO it later).
        start_ids = [oid for oid in bounds if bounds[oid][0] > 0]
    
        # per-signature frame indices actually used, included in the
        # report as `signature_frames` so a suspicious similarity can be
        # checked by hand.
        signature_frames: dict[str, dict] = {"end": {}, "start": {}, "pooled_groups": {}}
    
        try:
            end_sigs = {}
            for obj_id in end_ids:
                first, last = bounds[obj_id]
                # search BACKWARD from `last` (the risky edge, right before
                # the track goes empty) toward `first` -- ambiguous/empty
                # frames near the edge are skipped, extending the search
                # further into the track's more stable middle instead of
                # forcing a fixed-size window (see _sample_signature).
                candidates = list(range(last, first - 1, -1))
                emb, hist, frames_used = self._sample_signature(cap, mask_paths[obj_id], candidates, signature_samples, embedder)
                end_sigs[obj_id] = (emb, hist)
                signature_frames["end"][obj_id] = frames_used
    
            start_sigs = {}
            for obj_id in start_ids:
                first, last = bounds[obj_id]
                # search FORWARD from `first` (the risky reappearance edge)
                # toward `last`, same reasoning as above.
                candidates = list(range(first, last + 1))
                emb, hist, frames_used = self._sample_signature(cap, mask_paths[obj_id], candidates, signature_samples, embedder)
                start_sigs[obj_id] = (emb, hist)
                signature_frames["start"][obj_id] = frames_used
        finally:
            cap.release()

        candidate_pairs = self._resolve_merges(end_ids, start_ids, bounds, end_sigs, start_sigs, merge_threshold)
        merges = [
            (c["from_id"], c["into_id"], c["similarity"])
            for c in candidate_pairs if c["accepted"]
        ]

        # Zeroth-pass pairs are the only legitimate same-time unions; every
        # other merge is applied through a temporal-consistency veto (see
        # `_group_chains_with_temporal_veto`). Order = trust: overlap merges
        # first, then pass 1 by descending similarity.
        allowed_overlap_pairs = {frozenset((r["id_a"], r["id_b"])) for r in overlap_accepted}
        pass1_sorted = sorted(merges, key=lambda m: -m[2])
        canonical, vetoed_merges = self._group_chains_with_temporal_veto(
            overlap_merge_tuples + pass1_sorted, all_ids, bounds, allowed_overlap_pairs,
            max_tolerated_overlap_frames=min_overlap_frames,
        )

        # --- second pass: pooled-group fallback for orphan start tracks
        # pass one couldn't match against any single fragment ---
        merged_start_ids = {c["into_id"] for c in candidate_pairs if c["accepted"]}
        orphan_start_ids = [s for s in start_ids if s not in merged_start_ids]

        group_candidates: list[dict] = []
        if orphan_start_ids:
            pass1_groups: dict[int, list[int]] = {}
            for oid in all_ids:
                pass1_groups.setdefault(canonical[oid], []).append(oid)
            # EVERY pass-one group is a candidate here, including a trivial
            # one-member group formed from another orphan -- excluding those
            # would also hide them as candidates for every OTHER orphan, not
            # just for themselves. Self-matching is already excluded below
            # via `if o in members: continue`.
            candidate_group_ids = list(pass1_groups.keys())

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise FileNotFoundError(f"Could not open video: {video_path}")
            try:
                group_sigs = {}
                for g in candidate_group_ids:
                    emb, hist, member_frames_used = self._pooled_group_signature(
                        cap, mask_paths, pass1_groups[g], bounds, pooled_samples_per_member, embedder,
                    )
                    group_sigs[g] = (emb, hist)
                    signature_frames["pooled_groups"][g] = member_frames_used
            finally:
                cap.release()
            orphan_sigs = {o: start_sigs[o] for o in orphan_start_ids}

            group_candidates = self._resolve_group_merges(
                orphan_start_ids,
                {g: pass1_groups[g] for g in candidate_group_ids},
                bounds, group_sigs, orphan_sigs, merge_threshold,
                orphan_groups={o: pass1_groups[canonical[o]] for o in orphan_start_ids},
            )
            extra_merges = [
                (min(pass1_groups[c["group_id"]]), c["orphan_id"], c["similarity"])
                for c in group_candidates if c["accepted"]
            ]
            if extra_merges:
                extra_sorted = sorted(extra_merges, key=lambda m: -m[2])
                canonical, vetoed_merges = self._group_chains_with_temporal_veto(
                    overlap_merge_tuples + pass1_sorted + extra_sorted, all_ids, bounds,
                    allowed_overlap_pairs, max_tolerated_overlap_frames=min_overlap_frames,
                )
        # Write the merged MaskDir: one file per distinct canonical id,
        # union (logical OR) of every member's mask. Streamed lock-step --
        # every member's file is already padded to the same total_frames
        # length, so frame t aligns across members with no seeking needed,
        # and nothing allocates a full (total_frames, height, width) array.
        out_dir_path = Path(out_mask_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        groups: dict[int, list[int]] = {}
        for oid in all_ids:
            groups.setdefault(canonical[oid], []).append(oid)

        # The report is keyed by canonical id (smallest original id of the
        # group), which leaves gaps (e.g. 0,2,4,6). Output FILENAMES get a
        # sequential id instead (0,1,2,3, same order), since consumers such
        # as `TrackingTool.visualize` label by filename; `output_id_remap`
        # in the report maps canonical id -> output id.
        sorted_canon_ids = sorted(groups.keys())
        output_id_remap = {canon: seq for seq, canon in enumerate(sorted_canon_ids)}

        with VideoReader(path=video_path) as video_reader:
            frame_rate = video_reader.frame_rate

        for canon_id, members in groups.items():
            member_caps = [cv2.VideoCapture(str(mask_paths[m])) for m in members]
            for member, mcap in zip(members, member_caps):
                if not mcap.isOpened():
                    for c in member_caps:
                        c.release()
                    raise FileNotFoundError(f"Could not open mask video: {mask_paths[member]}")

            out_path = out_dir_path / f"{output_id_remap[canon_id]}.mp4"

            writer = VideoWriter(
                out_path,
                input_dict={
                    "-r": str(frame_rate),
                },
                output_dict={
                    "-c:v": "libx264",
                    "-crf": "0",
                    "-pix_fmt": "yuv420p",
                },
                overwrite=self.overwrite,
            )

            try:
                for _t in range(total_frames):
                    merged_frame = np.zeros((height, width), dtype=bool)
                    for mcap in member_caps:
                        ok, raw = mcap.read()
                        if not ok:
                            continue  # this member's file ended early (shouldn't
                                    # happen given the padding invariant, but
                                    # degrade gracefully rather than crash)
                        gray = raw[:, :, 0] if raw.ndim == 3 else raw
                        merged_frame |= gray > DEFAULT_MASK_THRESHOLD
                    frame_rgb = np.repeat((merged_frame.astype(np.uint8) * 255)[..., np.newaxis], 3, axis=-1)
                    writer.write(frame_rgb)
            finally:
                writer.close()
                for mcap in member_caps:
                    mcap.release()

        report = {
            "source_mask_dir": str(mask_dir),
            "output_mask_dir": str(out_mask_dir),
            "total_frames": total_frames,
            "original_id_count": len(all_ids),
            "merged_id_count": len(groups),
            "excluded_as_too_short": excluded_short,
            "merge_threshold": merge_threshold,
            "min_fragment_frames": min_fragment_frames,
            "osnet_used": embedder is not None,
            "overlap_pixel_gap_threshold_px": overlap_pixel_gap_threshold,
            "overlap_centroid_threshold_px": overlap_centroid_threshold,
            "min_overlap_frames": min_overlap_frames,
            "overlap_classifier_used": classifier is not None,
            "overlap_classifier_path": overlap_classifier_path,
            "accepted_overlap_merges": overlap_accepted,
            "rejected_overlap_candidates": overlap_rejected,
            "accepted_merges": [
                {"from_id": e, "into_id": s, "similarity": round(sim, 3)}
                for e, s, sim in merges
            ],
            "rejected_candidates": [
                {"from_id": c["from_id"], "into_id": c["into_id"], "similarity": c["similarity"]}
                for c in candidate_pairs if not c["accepted"]
            ],
            "vetoed_merges": vetoed_merges,
            "pooled_group_samples_per_member": pooled_samples_per_member,
            "pooled_group_candidates": [
                {"orphan_id": c["orphan_id"], "group_id": c["group_id"],
                "similarity": c["similarity"], "accepted": c["accepted"]}
                for c in group_candidates
            ],
            "groups": {str(canon): members for canon, members in groups.items()},
            "output_id_remap": {str(canon): seq for canon, seq in output_id_remap.items()},
            "signature_frames": {
                "end": {str(oid): frames for oid, frames in signature_frames["end"].items()},
                "start": {str(oid): frames for oid, frames in signature_frames["start"].items()},
                "pooled_groups": {
                    str(gid): {str(mid): frames for mid, frames in member_frames.items()}
                    for gid, member_frames in signature_frames["pooled_groups"].items()
                },
            },
        }
        return report

# -------------------------------------------------------------------------------------------
# ----------------------------- Masks IO ----------------------------------------------------
# -------------------------------------------------------------------------------------------

    def _list_mask_files(self, mask_dir: str) -> dict[int, Path]:
        """`{id: path}` for every `<id>.mp4` in `mask_dir`, without decoding
        anything."""
        dir_path = Path(mask_dir)
        paths: dict[int, Path] = {}
        if not dir_path.is_dir():
            return paths
        for p in dir_path.iterdir():
            m = _MASK_FILENAME_RE.match(p.name)
            if m:
                paths[int(m.group(1))] = p
        return paths


    def _scan_track(
        self,
        mask_path: Path,
        threshold: int = DEFAULT_MASK_THRESHOLD,
    ) -> tuple[int, int, int, int, int, int] | None:
        """Streams one id's mask video once (O(1) memory) and returns
        `(first, last, real_frames, decoded_frame_count, height, width)`,
        or `None` if the track has no non-empty frame at all.
        `decoded_frame_count` is measured by actually walking the file
        rather than trusting `cv2.CAP_PROP_FRAME_COUNT` metadata, which has
        disagreed with the real count in practice."""
        cap = cv2.VideoCapture(str(mask_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open mask video: {mask_path}")

        first = last = None
        real_frames = 0
        decoded = 0
        height = width = None
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if height is None:
                    height, width = frame.shape[0], frame.shape[1]
                gray = frame[:, :, 0] if frame.ndim == 3 else frame
                if bool((gray > threshold).any()):
                    if first is None:
                        first = decoded
                    last = decoded
                    real_frames += 1
                decoded += 1
        finally:
            cap.release()

        if first is None:
            return None
        return first, last, real_frames, decoded, height, width


    def _read_mask_frame(
        self,
        cap: cv2.VideoCapture,
        frame_idx: int,
        threshold: int = DEFAULT_MASK_THRESHOLD,
    ) -> np.ndarray | None:
        """Seeks an already-open mask-file `VideoCapture` to `frame_idx` and
        returns the decoded boolean mask frame, or `None` on a failed
        seek/read (past end of file, corrupt frame)."""
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, raw = cap.read()
        if not ok:
            return None
        gray = raw[:, :, 0] if raw.ndim == 3 else raw
        return gray > threshold

# -------------------------------------------------------------------------------------------
# ----------------------------- Frame signature building ------------------------------------
# -------------------------------------------------------------------------------------------

    def _frame_signal(
        self,
        cap: cv2.VideoCapture,
        mask_frame: np.ndarray,
        frame_idx: int,
        embedder: OSNetEmbedder | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """OSNet embedding + hue histogram for ONE frame, or `(None, None)`
        if the frame is empty or its mask isn't a single, non-border-touching
        component."""
        if not mask_frame.any():
            return None, None
        poly = self._mask_to_polygon_single_component(mask_frame)
        if poly.shape[0] < 3:
            return None, None
        height, width = mask_frame.shape[:2]
        if self._touches_border(poly, height, width):
            return None, None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            return None, None
        box = self._polygon_to_box(poly)
        embedding = None
        if embedder is not None:
            embedding = embedder.embed(frame, box, poly=poly)
        histogram = self._mask_hue_histogram(frame, poly)
        return embedding, histogram


    def _finalize_signature(
        self,
        embeddings: list[np.ndarray],
        histograms: list[np.ndarray],
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Averages collected per-frame embeddings/histograms into one
        signature (re-normalized after averaging). `None` for whichever
        signal had no usable frames at all."""
        embedding = None
        if embeddings:
            embedding = np.mean(embeddings, axis=0)
            norm = np.linalg.norm(embedding)
            embedding = embedding / norm if norm > 1e-9 else None

        histogram = None
        if histograms:
            histogram = np.mean(histograms, axis=0)
            total = histogram.sum()
            histogram = histogram / total if total > 1e-9 else None

        return embedding, histogram


    def _sample_signature(
        self,
        cap: cv2.VideoCapture,
        mask_path: Path,
        candidate_frames: list[int],
        signature_samples: int,
        embedder: OSNetEmbedder | None,
        threshold: int = DEFAULT_MASK_THRESHOLD,
    ) -> tuple[np.ndarray | None, np.ndarray | None, list[int]]:
        """Averages OSNet embedding + hue histogram over up to
        `signature_samples` usable frames drawn from `candidate_frames`
        (see `_frame_signal`), stopping once that many are collected. Also
        returns the frame indices actually used, so a suspicious similarity
        score can be checked by opening those exact frames.

        `candidate_frames` must be ordered from the track's risky edge
        OUTWARD (backward from the last frame for an ending track, forward
        from the first for a starting one), so skipped ambiguous frames
        naturally extend the search into the track's more stable middle."""
        embeddings: list[np.ndarray] = []
        histograms: list[np.ndarray] = []
        used_frame_indices: list[int] = []

        mask_cap = cv2.VideoCapture(str(mask_path))
        if not mask_cap.isOpened():
            raise FileNotFoundError(f"Could not open mask video: {mask_path}")

        used_frames = 0
        try:
            for frame_idx in candidate_frames:
                if used_frames >= signature_samples:
                    break
                if frame_idx < 0:
                    continue
                mask_frame = self._read_mask_frame(mask_cap, frame_idx, threshold)
                if mask_frame is None:
                    continue  # past this track's own file bounds -- skip, don't guess
                emb, hist = self._frame_signal(cap, mask_frame, frame_idx, embedder)
                if emb is None and hist is None:
                    continue  # empty, ambiguous, or border-touching -- skip, don't guess
                if emb is not None:
                    embeddings.append(emb)
                if hist is not None:
                    histograms.append(hist)
                used_frames += 1
                used_frame_indices.append(frame_idx)
        finally:
            mask_cap.release()

        embedding, histogram = self._finalize_signature(embeddings, histograms)
        return embedding, histogram, used_frame_indices

    def _evenly_spaced_frames(self,first: int, last: int, n: int) -> list[int]:
        """`n` frame indices evenly spread across `[first, last]` -- used by
        `_pooled_group_signature` to sample pose/lighting diversity across a
        fragment's whole span, unlike `_sample_signature`'s edge-anchored
        order."""
        span = last - first + 1
        if span <= n:
            return list(range(first, last + 1))
        if n <= 1:
            return [first]
        return sorted({int(round(first + i * (span - 1) / (n - 1))) for i in range(n)})


    def _search_nearby_signal(
        self,
        cap: cv2.VideoCapture,
        member_cap: cv2.VideoCapture,
        target: int,
        first: int,
        last: int,
        max_radius: int,
        exclude: set[int],
        embedder: OSNetEmbedder | None,
        threshold: int = DEFAULT_MASK_THRESHOLD,
    ) -> tuple[int, np.ndarray | None, np.ndarray | None] | None:
        """Tries `target`, then alternately expands outward (+1, -1, +2, -2,
        ...) up to `max_radius`, clamped to `[first, last]` and skipping
        `exclude`, returning the first frame that passes `_frame_signal`'s
        checks -- so one bad frame at an evenly-spaced target position
        doesn't lose that sampling slot entirely."""
        offsets = [0]
        for r in range(1, max_radius + 1):
            offsets.append(r)
            offsets.append(-r)
        for offset in offsets:
            frame_idx = target + offset
            if frame_idx < first or frame_idx > last or frame_idx in exclude:
                continue
            mask_frame = self._read_mask_frame(member_cap, frame_idx, threshold)
            if mask_frame is None:
                continue
            emb, hist = self._frame_signal(cap, mask_frame, frame_idx, embedder)
            if emb is None and hist is None:
                continue
            return frame_idx, emb, hist
        return None


    def _pooled_group_signature(
        self,
        cap: cv2.VideoCapture,
        mask_paths: dict[int, Path],
        member_ids: list[int],
        bounds: dict[int, tuple[int, int]],
        samples_per_member: int,
        embedder: OSNetEmbedder | None,
        threshold: int = DEFAULT_MASK_THRESHOLD,
    ) -> tuple[np.ndarray | None, np.ndarray | None, dict[int, list[int]]]:
        """Aggregated appearance signature for an already-confirmed GROUP of
        ids, pooling clean frames spread across ALL members (different
        times/poses/lighting) instead of just one fragment's edge frames --
        the pass-two fallback for an orphan start track pass one couldn't
        match. For each evenly-spaced target frame per member, searches a
        small neighborhood (`_search_nearby_signal`) instead of only the
        exact target, so one unlucky frame doesn't cost that member a whole
        sample. Also returns `{member_id: [frame indices used]}` for the
        report."""
        embeddings: list[np.ndarray] = []
        histograms: list[np.ndarray] = []
        used_frame_indices: dict[int, list[int]] = {}
        for member_id in member_ids:
            if member_id not in bounds or member_id not in mask_paths:
                continue  # e.g. a too-short fragment folded into this group
            first, last = bounds[member_id]
            targets = self._evenly_spaced_frames(first, last, samples_per_member)
            # search radius per target: roughly half the average spacing
            # between targets, capped by MAX_POOLED_SEARCH_RADIUS.
            avg_gap = max(1, (last - first + 1) // max(1, samples_per_member))
            max_radius = min(max(1, avg_gap // 2), MAX_POOLED_SEARCH_RADIUS)

            member_cap = cv2.VideoCapture(str(mask_paths[member_id]))
            if not member_cap.isOpened():
                raise FileNotFoundError(f"Could not open mask video: {mask_paths[member_id]}")
            try:
                used: set[int] = set()
                for target in targets:
                    found = self._search_nearby_signal(
                        cap, member_cap, target, first, last, max_radius, used, embedder, threshold,
                    )
                    if found is None:
                        continue  # nothing usable near this slot -- some signal beats none
                    frame_idx, emb, hist = found
                    used.add(frame_idx)
                    if emb is not None:
                        embeddings.append(emb)
                    if hist is not None:
                        histograms.append(hist)
                    used_frame_indices.setdefault(member_id, []).append(frame_idx)
            finally:
                member_cap.release()
        embedding, histogram = self._finalize_signature(embeddings, histograms)
        return embedding, histogram, used_frame_indices


    def _pair_similarity(
        self,
        end_sig: tuple[np.ndarray | None, np.ndarray | None],
        start_sig: tuple[np.ndarray | None, np.ndarray | None],
    ) -> float:
        """0..1 similarity between two signatures: the STRONGEST of (OSNet,
        color), never their sum/average."""
        end_emb, end_hist = end_sig
        start_emb, start_hist = start_sig
        scores: list[float] = []
        emb_sim = self.embedding_similarity(end_emb, start_emb)
        if emb_sim is not None:
            scores.append(emb_sim)
        if end_hist is not None and start_hist is not None:
            scores.append(self._histogram_similarity(end_hist, start_hist))
        return max(scores) if scores else 0.0


# -------------------------------------------------------------------------------------------
# ----------------------------- Color / Hue Histogram Utilities -----------------------------
# -------------------------------------------------------------------------------------------

    def _polygon_to_box(self, poly: np.ndarray) -> np.ndarray:
        """Bounding box (x1,y1,x2,y2) of the polygon. `[0,0,0,0]` if empty."""
        if poly.shape[0] == 0:
            return np.zeros(4)
        x1, y1 = poly.min(axis=0)
        x2, y2 = poly.max(axis=0)
        return np.array([x1, y1, x2, y2], dtype=float)


    def _mask_hue_histogram(self,frame: np.ndarray, poly: np.ndarray,
                            bins: int = _HUE_HIST_BINS) -> np.ndarray | None:
        """Hue histogram (OpenCV Hue, 0-179) of the pixels inside the mask
        polygon, weighted by saturation and normalized to sum 1. Captures a
        two-tone/striped garment as two peaks, unlike a single average hue.
        `None` if the polygon is empty/degenerate or too few trustworthy
        (non-desaturated) pixels remain."""
        if poly.shape[0] < 3:
            return None
        h, w = frame.shape[:2]
        pts = np.round(poly).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        if cv2.countNonZero(mask) < 25:
            return None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ys, xs = np.where(mask > 0)
        hue = hsv[ys, xs, 0].astype(np.float64)          # 0..179
        sat = hsv[ys, xs, 1].astype(np.float64) / 255.0  # 0..1, used as weight

        valid = sat > 0.15  # nearly gray pixels: their hue is sensor noise
        if valid.sum() < 25:
            return None
        hue, sat = hue[valid], sat[valid]

        hist, _ = np.histogram(hue, bins=bins, range=(0, 180), weights=sat)
        total = hist.sum()
        if total <= 0:
            return None
        return hist / total


    def _histogram_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """0..1 similarity between two normalized hue histograms via
        intersection (sum(min(a, b))): 1.0 identical, 0.0 no overlap."""
        return float(np.clip(np.minimum(a, b).sum(), 0.0, 1.0))


    def _mask_to_polygon_single_component(self, mask_frame: np.ndarray) -> np.ndarray:
        """Polygon of the mask frame's connected component, but ONLY if
        there's EXACTLY ONE region (empty `(0,2)` polygon otherwise, e.g. a
        stray second blob makes the frame too ambiguous to trust)."""
        mask_u8 = mask_frame.astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 1:
            return np.empty((0, 2))
        return contours[0].reshape(-1, 2).astype(float)


    def _touches_border(self, poly: np.ndarray, height: int, width: int, margin: int = 1) -> bool:
        """True if `poly`'s bounding box comes within `margin` px of the
        frame edge -- used to reject a single-component frame whose one
        region is actually a stray artifact stuck to the border rather than
        the tracked person."""
        if poly.shape[0] == 0:
            return False
        x_min, y_min = poly.min(axis=0)
        x_max, y_max = poly.max(axis=0)
        return bool(
            x_min <= margin or y_min <= margin
            or x_max >= width - 1 - margin or y_max >= height - 1 - margin
        )


    def _mask_centroid(self, mask: np.ndarray) -> tuple[float, float]:
        """Pixel centroid `(x, y)` of a boolean mask (caller ensures `mask.any()`)."""
        ys, xs = np.where(mask)
        return float(xs.mean()), float(ys.mean())


    def _centroid_distance(self, mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        """Euclidean distance between two masks' centroids."""
        ax, ay = self._mask_centroid(mask_a)
        bx, by = self._mask_centroid(mask_b)
        return float(np.hypot(ax - bx, ay - by))


    def _mask_bbox_diagonal(self, mask: np.ndarray) -> float:
        """Diagonal (px) of a mask's tight bounding box -- a cheap proxy for
        "how big is this fragment on screen right now", used to normalize a
        raw pixel distance into a scale-portable ratio."""
        ys, xs = np.where(mask)
        if ys.size == 0:
            return 0.0
        h = float(ys.max() - ys.min())
        w = float(xs.max() - xs.min())
        return float(np.hypot(h, w))


    def _mask_min_pixel_distance(self, mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        """Minimum real pixel-to-pixel distance between two masks'
        silhouettes (0.0 if they touch/overlap) -- via `cv2.distanceTransform`
        on the complement of `mask_a`, read off at `mask_b`'s pixels.
        O(H*W), and unlike a bounding-box gap, doesn't false-positive when
        two boxes touch but the actual silhouettes are far apart."""
        inv_a = (~mask_a).astype(np.uint8)
        dist_map = cv2.distanceTransform(inv_a, cv2.DIST_L2, 5)
        return float(dist_map[mask_b].min())

# -------------------------------------------------------------------------------------------
# ----------------------------- OSNet embedding Utilities -----------------------------------
# -------------------------------------------------------------------------------------------


    def embedding_similarity(self,a: np.ndarray | None, b: np.ndarray | None) -> float | None:
        """0..1 similarity between two L2-normalized embeddings, via cosine
        similarity rescaled from [-1, 1] to [0, 1] (same 0..1 convention as
        the module's other signals, not cosine's native [-1, 1] convention).
        `None` if either embedding is missing."""
        if a is None or b is None:
            return None
        cos = float(np.dot(a, b))
        return float(np.clip((cos + 1.0) / 2.0, 0.0, 1.0))


    def torchreid_available(self) -> bool:
        """True if `torchreid.utils.FeatureExtractor` is actually reachable
        (same check as `_resolve_feature_extractor`, not just a bare
        `import torchreid`, which can succeed even when that attribute
        isn't reachable -- see that function's docstring). Doesn't
        instantiate a model or touch the GPU. For availability checks only:
        if OSNet is explicitly requested but unavailable,
        `OSNetEmbedder.__init__` still raises with install instructions
        rather than silently skipping the signal."""
        try:
            _resolve_feature_extractor()
        except (ImportError, AttributeError):
            return False
        except Exception as exc:
            # torchreid is unmaintained and can fail on import with something
            # other than ImportError on newer numpy/torch (e.g. np.float/
            # np.int removed in numpy>=1.24, still referenced by some
            # torchreid/yacs versions -> AttributeError instead). Caught
            # here and reported, rather than left to propagate and look like
            # an unrelated crash.
            print(f"[appearance_embedding] torchreid installed but the import fails "
                f"({type(exc).__name__}: {exc}) -- OSNet embedding not available. "
                f"Likely a version incompatibility (torchreid is a package "
                f"no longer actively maintained, often conflicting with recent "
                f"numpy/torch).")
            return False
        return True

# -------------------------------------------------------------------------------------------
# ----------------------------- Masks merging -----------------------------------------------
# -------------------------------------------------------------------------------------------


    def _resolve_merges(
        self,
        end_ids: list[int],
        start_ids: list[int],
        bounds: dict[int, tuple[int, int]],
        end_sigs: dict[int, tuple],
        start_sigs: dict[int, tuple],
        merge_threshold: float,
        blacklist: set[tuple[int, int]] | None = None,
    ) -> list[dict]:
        """Pass 1: global Hungarian assignment between `end_ids` (rows) and
        `start_ids` (columns). Returns every temporally-valid pair Hungarian
        assigned (end strictly before start), each tagged
        `"accepted": similarity >= merge_threshold` -- including near-misses,
        useful when tuning the threshold. A single global assignment (not
        pairwise greedy) so two simultaneous fragmentation events don't
        steal each other's correct match.

        `blacklist` is a set of `(end_id, start_id)` pairs forced to
        `_IMPOSSIBLE_COST` -- used by the caller to retry the assignment
        after a temporal veto, so the vetoed edge can't win the same slot
        twice and the next-best (possibly temporally-sensible) candidate
        gets a chance instead."""
        if not end_ids or not start_ids:
            return []

        blacklist = blacklist or set()

        cost = np.full((len(end_ids), len(start_ids)), _IMPOSSIBLE_COST)
        for i, e in enumerate(end_ids):
            for j, s in enumerate(start_ids):
                if e == s or bounds[e][1] >= bounds[s][0] or (e, s) in blacklist:
                    continue  # same track, start doesn't come after end, or vetoed earlier
                sim = self._pair_similarity(end_sigs[e], start_sigs[s])
                cost[i, j] = 1.0 - sim

        row_idx, col_idx = linear_sum_assignment(cost)
        candidates = []
        for r, c in zip(row_idx, col_idx):
            if cost[r, c] >= _IMPOSSIBLE_COST:
                continue  # not a real candidate -- see docstring
            similarity = float(1.0 - cost[r, c])
            candidates.append({
                "from_id": end_ids[r],
                "into_id": start_ids[c],
                "similarity": round(similarity, 3),
                "accepted": bool(similarity >= merge_threshold),
            })
        return candidates


    def _resolve_group_merges(
        self,
        orphan_ids: list[int],
        groups: dict[int, list[int]],
        bounds: dict[int, tuple[int, int]],
        group_sigs: dict[int, tuple],
        orphan_sigs: dict[int, tuple],
        merge_threshold: float,
        orphan_groups: dict[int, list[int]] | None = None,
        blacklist: set[tuple[int, int]] | None = None,
    ) -> list[dict]:
        """Pass 2: global Hungarian assignment between orphan start tracks
        (rows -- ones pass one left unmatched) and candidate GROUPS
        (columns, `{canonical_id: member_ids}` from pass one). A group only
        qualifies for an orphan if NO member overlaps the orphan in time (a
        real person can't be two simultaneous tracks) and at least one
        member genuinely ends before the orphan starts. Same global-
        assignment, "tag every real candidate" conventions as
        `_resolve_merges`.

        `orphan_groups` (`{orphan_id: member_ids}`) is the orphan's OWN
        pass-one group: an orphan is unmatched on its START side but may
        already have been merged on its END side (x -> y), so accepting
        "orphan into group G" really merges its whole chain into G. The
        overlap check therefore runs on every member of the orphan's chain,
        not just the orphan -- otherwise a 30-frame fragment can bridge two
        people who coexist for the entire session.

        `blacklist` is a set of `(orphan_id, group_id)` pairs forced to
        `_IMPOSSIBLE_COST`, same veto-retry purpose as in `_resolve_merges`."""
        if not orphan_ids or not groups:
            return []

        blacklist = blacklist or set()

        group_ids = list(groups.keys())
        cost = np.full((len(orphan_ids), len(group_ids)), _IMPOSSIBLE_COST)
        for i, o in enumerate(orphan_ids):
            o_first, o_last = bounds[o]
            chain = (orphan_groups or {}).get(o, [o])
            chain_bounds = [bounds[m] for m in chain if m in bounds] or [bounds[o]]
            for j, g in enumerate(group_ids):
                members = groups[g]
                if o in members or (o, g) in blacklist:
                    continue  # orphan is (trivially) already part of this group, or vetoed earlier
                member_bounds = [bounds[m] for m in members if m in bounds]
                if not member_bounds:
                    continue  # e.g. every member was too short to have bounds
                if any(m_first <= c_last and m_last >= c_first
                    for m_first, m_last in member_bounds
                    for c_first, c_last in chain_bounds):
                    continue  # a member of this group is active at the same time as the orphan's chain -- can't be the same person
                if not any(m_last < o_first for _m_first, m_last in member_bounds):
                    continue  # no member of this group actually ends before the orphan starts
                sim = self._pair_similarity(group_sigs[g], orphan_sigs[o])
                cost[i, j] = 1.0 - sim

        row_idx, col_idx = linear_sum_assignment(cost)
        candidates = []
        for r, c in zip(row_idx, col_idx):
            if cost[r, c] >= _IMPOSSIBLE_COST:
                continue  # not a real candidate -- see docstring
            similarity = float(1.0 - cost[r, c])
            candidates.append({
                "orphan_id": orphan_ids[r],
                "group_id": group_ids[c],
                "similarity": round(similarity, 3),
                "accepted": bool(similarity >= merge_threshold),
            })
        return candidates


    def _group_chains(self, merges: list[tuple[int, int, float]], all_ids: list[int]) -> dict[int, int]:
        """Turns a list of accepted (end_id -> start_id) merges into
        `{original_id: canonical_id}` via union-find, following chains (A
        merges into B, B merges into C => all map to the same id). The
        canonical id is the group's SMALLEST original id -- a stable,
        deterministic output filename, no other meaning."""
        parent = {oid: oid for oid in all_ids}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        for end_id, start_id, _sim in merges:
            union(end_id, start_id)

        return {oid: find(oid) for oid in all_ids}


    def _overlap_frames(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        """Number of frames where both `(first, last)` spans are active
        (0 when they don't overlap)."""
        return max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)


    def _group_chains_with_temporal_veto(
        self,
        merges: list[tuple[int, int, float]],
        all_ids: list[int],
        bounds: dict[int, tuple[int, int]],
        allowed_overlap_pairs: set[frozenset[int]],
        max_tolerated_overlap_frames: int = 0,
    ) -> tuple[dict[int, int], list[dict]]:
        """Same union-find as `_group_chains`, but every union is applied
        ONLY if the resulting group stays temporally consistent: a real
        person can't be two tracks alive at the same time, so a merge that
        would put two simultaneously-active ids into the same group is
        vetoed -- unless that exact pair was accepted by the zeroth pass
        (`allowed_overlap_pairs`), which is the one legitimate case of
        same-body simultaneous fragments (a garment being put on/taken
        off). Overlaps of at most `max_tolerated_overlap_frames` are
        ignored (tracker jitter at a fragment boundary, not real
        coexistence).

        This is the whole-video counterpart of the zeroth pass's
        one-match-per-fragment rule: pass 1/2 only ever check the two ids
        of a pair against each other, so a short ambiguous fragment can
        still bridge two different people by transitivity (A->x and x->B
        both look plausible even though A and B coexist for thousands of
        frames). Merges are applied in the order given -- put the most
        trusted ones first, they win any conflict.

        Returns `({original_id: canonical_id}, vetoed)` where `vetoed`
        lists every merge skipped, with the conflicting pair and its
        overlap length, for the report."""
        parent = {oid: oid for oid in all_ids}
        members: dict[int, list[int]] = {oid: [oid] for oid in all_ids}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        vetoed: list[dict] = []
        for end_id, start_id, sim in merges:
            ra, rb = find(end_id), find(start_id)
            if ra == rb:
                continue
            conflict = None
            for a in members[ra]:
                if a not in bounds:
                    continue
                for b in members[rb]:
                    if b not in bounds or frozenset((a, b)) in allowed_overlap_pairs:
                        continue
                    ov = self._overlap_frames(bounds[a], bounds[b])
                    if ov > max_tolerated_overlap_frames:
                        conflict = (a, b, ov)
                        break
                if conflict:
                    break
            if conflict:
                vetoed.append({
                    "from_id": end_id, "into_id": start_id, "similarity": round(sim, 3),
                    "conflict_ids": [conflict[0], conflict[1]], "conflict_overlap_frames": conflict[2],
                })
                continue
            keep, drop = min(ra, rb), max(ra, rb)
            parent[drop] = keep
            members[keep].extend(members.pop(drop))

        return {oid: find(oid) for oid in all_ids}, vetoed

# -------------------------------------------------------------------------------------------
# ----------------------------- Overlap resolution ------------------------------------------
# -------------------------------------------------------------------------------------------

    def _stream_overlap_stats(
        self,
        path_a: Path,
        path_b: Path,
        start_frame: int,
        end_frame: int,
        threshold: int = DEFAULT_MASK_THRESHOLD,
    ) -> tuple[list[float], list[float], list[float]]:
        """Streams both mask files over `[start_frame, end_frame]` and, for
        every frame where BOTH have content, computes the real pixel-to-
        pixel minimum distance, the centroid distance, and a scale
        reference (mean bbox diagonal). Returns the three lists in frame
        order."""
        cap_a = cv2.VideoCapture(str(path_a))
        cap_b = cv2.VideoCapture(str(path_b))
        if start_frame > 0:
            cap_a.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            cap_b.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        pixel_dists: list[float] = []
        centroid_dists: list[float] = []
        scales: list[float] = []
        try:
            for _f in range(start_frame, end_frame + 1):
                ok_a, ra = cap_a.read()
                ok_b, rb = cap_b.read()
                if not ok_a or not ok_b:
                    break
                ga = ra[:, :, 0] if ra.ndim == 3 else ra
                gb = rb[:, :, 0] if rb.ndim == 3 else rb
                ma = ga > threshold
                mb = gb > threshold
                if not ma.any() or not mb.any():
                    continue
                pixel_dists.append(self._mask_min_pixel_distance(ma, mb))
                centroid_dists.append(self._centroid_distance(ma, mb))
                scales.append((self._mask_bbox_diagonal(ma) + self._mask_bbox_diagonal(mb)) / 2.0)
        finally:
            cap_a.release()
            cap_b.release()

        return pixel_dists, centroid_dists, scales


    def _classifier_score(self,classifier: dict, feature_values: dict[str, float]) -> float:
        """`sigmoid(bias + sum(weight_i * feature_i))` for the learned
        overlap classifier. `classifier` is the JSON produced by
        `train_overlap_classifier.py`: `{"features": [...], "weights": [...],
        "bias": ...}`."""
        z = float(classifier["bias"])
        for name, w in zip(classifier["features"], classifier["weights"]):
            z += float(w) * feature_values[name]
        return float(1.0 / (1.0 + np.exp(-z)))

    def _resolve_overlap_merges(
        self,
        mask_paths: dict[int, Path],
        bounds: dict[int, tuple[int, int]],
        pixel_gap_threshold: float,
        centroid_threshold: float,
        min_overlap_frames: int,
        mask_threshold: int = DEFAULT_MASK_THRESHOLD,
        classifier: dict | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """For every pair of ids that overlap in time and both have
        trustworthy bounds: stream both mask files over their shared
        window, and if enough both-active frames exist, compute the MEDIAN
        pixel and centroid distance. Accept as "same body, split into
        simultaneous fragments" only if both medians clear their threshold
        (pixel distance alone can't tell a same-body split from a brief
        occlusion between two different people; centroid distance is what
        actually separates the two classes in the calibration data).

        If `classifier` is given, it REPLACES the two fixed thresholds: a
        pair qualifies if the model's probability >= 0.5, ranked by that
        probability instead of raw centroid distance.

        Returns `(accepted, rejected)`, recording both raw medians and the
        both-active frame count for every candidate considered, not just
        accepted ones.

        IMPORTANT: each id is accepted into at most one pair (its single
        closest match). A small static object mistakenly given its own SAM
        track (e.g. a jacket on a radiator) can clear both thresholds
        against TWO different real people who happen to sit near it at
        different times -- accepting both pairs would union those two
        people together by transitivity, even though comparing them
        directly correctly rejects the match. Restricting each id to its
        nearest match (greedy, closest pairs claimed first) prevents this
        without needing to know in advance which candidate is the false one."""
        ids = sorted(bounds.keys())
        rejected: list[dict] = []

        # first pass: collect every pair that clears both thresholds as a
        # QUALIFYING candidate (not yet accepted -- greedy matching below
        # decides who actually gets it).
        qualifying: list[dict] = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                a_first, a_last = bounds[a]
                b_first, b_last = bounds[b]
                overlap_start = max(a_first, b_first)
                overlap_end = min(a_last, b_last)
                if overlap_start > overlap_end:
                    continue  # no temporal overlap at all -- pass 1/2 already handle this pair

                pixel_dists, centroid_dists, scales = self._stream_overlap_stats(
                    mask_paths[a], mask_paths[b], overlap_start, overlap_end, mask_threshold,
                )
                n = len(pixel_dists)
                if n < min_overlap_frames:
                    continue  # too few both-active frames to trust a median

                median_pixel = float(np.median(pixel_dists))
                median_centroid = float(np.median(centroid_dists))
                median_scale = float(np.median(scales))
                # normalized = centroid distance as a fraction of on-screen
                # scale right now -- portable across cameras/zoom levels.
                median_centroid_norm = median_centroid / median_scale if median_scale > 0 else float("inf")
                record = {
                    "id_a": a,
                    "id_b": b,
                    "median_pixel_gap_px": round(median_pixel, 1),
                    "median_centroid_dist_px": round(median_centroid, 1),
                    "median_scale_px": round(median_scale, 1),
                    "median_centroid_dist_norm": round(median_centroid_norm, 4),
                    "overlap_frames": n,
                }
                if classifier is not None:
                    score = self._classifier_score(classifier, {
                        "centroid_dist_norm": median_centroid_norm,
                        "pixel_gap_px": median_pixel,
                        "centroid_dist_px": median_centroid,
                    })
                    record["classifier_score"] = round(score, 4)
                    if score >= 0.5:
                        qualifying.append(record)
                    else:
                        rejected.append(record)
                elif median_pixel <= pixel_gap_threshold and median_centroid <= centroid_threshold:
                    qualifying.append(record)
                else:
                    rejected.append(record)

        # second pass: greedy nearest-first (or most-confident-first, with a
        # classifier) matching -- the strongest pairs claim their ids first,
        # so a fragment already claimed can't bridge a second, weaker match
        # (see docstring above). A qualifying pair that loses out this way
        # moves to `rejected` tagged `lost_to_closer_match`, kept visible
        # rather than silently dropped.
        if classifier is not None:
            qualifying.sort(key=lambda r: -r["classifier_score"])
        else:
            qualifying.sort(key=lambda r: r["median_centroid_dist_px"])
        claimed: set[int] = set()
        accepted: list[dict] = []
        for record in qualifying:
            a, b = record["id_a"], record["id_b"]
            if a in claimed or b in claimed:
                rejected.append({**record, "lost_to_closer_match": True})
                continue
            accepted.append(record)
            claimed.add(a)
            claimed.add(b)

        return accepted, rejected
