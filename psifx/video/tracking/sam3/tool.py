"""sam3 tracking tool."""

from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam3VideoModel, Sam3VideoProcessor

from psifx.io.video import VideoReader, VideoWriter
from psifx.utils.constants import SAM3_PATH
from psifx.video.tracking.tool import TrackingTool


class Sam3TrackingTool(TrackingTool):
    def __init__(
        self,
        device: str = "cpu",
        model_path: str = SAM3_PATH,
        api_token: str = None,
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
        # Keep raw video frames on CPU to cap CUDA memory usage for long clips.
        self.video_storage_device = "cpu" if self.device == "cuda" else self.device

        if self.verbose:
            print(f"Loading SAM3 model from '{self.model_path}' on device '{self.device}'")

        try:
            self.model = Sam3VideoModel.from_pretrained(self.model_path, token=api_token).to(
                self.device, dtype=self.compute_dtype
            )
            self.processor = Sam3VideoProcessor.from_pretrained(self.model_path, token=api_token)
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
    ):
        """
        Perform text-based segmentation and tracking from a video file.

        :param video_path: Path to the input video.
        :param mask_dir: Path to the output mask directory.
        :param text_prompt: Text description of objects to track.
        :param chunk_size: Number of frames to process at once.
        :param iou_threshold: IoU threshold for stitching chunks together.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}.")

        mask_dir = Path(mask_dir)
        if mask_dir.exists() and any(mask_dir.iterdir()):
            if self.overwrite:
                print(f"Mask directory {mask_dir} is non-empty")
            else:
                raise FileExistsError(f"Mask directory {mask_dir} is non-empty.")

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

    @staticmethod
    def _iter_video_chunks(
        video_path: Union[str, Path], chunk_size: int
    ) -> Iterable[Tuple[int, List[Image.Image]]]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video for reading: {video_path}")

        chunk: List[Image.Image] = []
        start_frame = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                chunk.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                if len(chunk) >= chunk_size:
                    yield start_frame, chunk
                    start_frame += len(chunk)
                    chunk = []

            if chunk:
                yield start_frame, chunk
        finally:
            cap.release()

    def _segment_chunk(self, chunk: List[Image.Image], text_prompt: str):
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
            for out in self.model.propagate_in_video_iterator(session, max_frame_num_to_track=len(chunk)):
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
                    id_mapping[obj_id] = next_global_id
                    next_global_id += 1

        return id_mapping, next_global_id

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
