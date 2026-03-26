"""sam3 tracking tool."""

import numpy as np
import torch
from PIL import Image
import cv2
from psifx.utils.constants import SAM3_PATH


from transformers import Sam3VideoModel, Sam3VideoProcessor

from typing import Union, Optional
from pathlib import Path

from psifx.io.video import VideoReader, VideoWriter
from psifx.video.tracking.tool import TrackingTool


class Sam3TrackingTool(TrackingTool):
    def __init__(self,
                 device: str = "cpu",
                 overwrite: bool = False,
                 verbose: Union[bool, int] = True,
                 ):
        super().__init__(
            device=device,
            overwrite=overwrite,
            verbose=verbose,
        )

        self.model = Sam3VideoModel.from_pretrained(SAM3_PATH).to(device, dtype=torch.bfloat16)
        self.processor = Sam3VideoProcessor.from_pretrained(SAM3_PATH)

    def infer(self,
              video_path: Union[str, Path],
              mask_dir: Union[str, Path],
              text_prompt: str = 'people',
              chunk_size: int = 300,
              iou_threshold: float = 0.3):
        """
        Perform text-based segmentation and tracking from a video file.

        :param video_path: Path to the input video.
        :param mask_dir: Path to the output mask directory.
        :param text_prompt: Text description of objects to track.
        :param chunk_size: Number of frames to process at once.
        :param iou_threshold: IoU threshold for stitching chunks together.
        """
        mask_dir = Path(mask_dir)
        if mask_dir.exists() and any(mask_dir.iterdir()):
            if self.overwrite:
                print(f"Mask directory {mask_dir} is non-empty")
            else:
                raise FileExistsError(f"Mask directory {mask_dir} is non-empty.")

        mask_dir.mkdir(parents=True, exist_ok=True)

        # Load video frames
        frames = self._load_video_frames(video_path)

        # Get frame rate for output
        with VideoReader(path=video_path) as video_reader:
            frame_rate = video_reader.frame_rate

        # Segment video in chunks
        chunked_outputs = self._segment_video(frames, text_prompt, chunk_size)

        # Stitch chunks with consistent IDs
        id_mappings = self._stitch_chunks(chunked_outputs, iou_threshold)

        # Build unified output
        unified = self._build_unified_output(chunked_outputs, id_mappings)

        # Write masks to separate videos
        self._write_masks(unified, mask_dir, frame_rate, frames[0].size)

    def _load_video_frames(self, video_path):
        """Load video frames as PIL Images."""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        return frames

    def _segment_video(self, frames, text_prompt, chunk_size):
        """Segment video in chunks."""
        chunked_outputs = []

        for start in range(0, len(frames), chunk_size):
            chunk = frames[start:start + chunk_size]

            session = self.processor.init_video_session(
                video=chunk,
                inference_device=self.device,
                processing_device=self.device,
                video_storage_device=self.device,
                dtype=torch.bfloat16,
            )
            self.processor.add_text_prompt(session, text_prompt)

            chunk_outputs = {
                'start_frame': start,
                'end_frame': start + len(chunk),
                'frames': {}
            }

            for out in self.model.propagate_in_video_iterator(session, max_frame_num_to_track=len(chunk)):
                chunk_outputs['frames'][out.frame_idx] = self.processor.postprocess_outputs(session, out)

            chunked_outputs.append(chunk_outputs)

            del session
            if self.device == "cuda":
                torch.cuda.empty_cache()

        return chunked_outputs

    @staticmethod
    def _compute_mask_iou(mask1, mask2):
        """Compute IoU between two binary masks."""
        intersection = (mask1 & mask2).sum()
        union = (mask1 | mask2).sum()
        return intersection / union if union > 0 else 0

    def _stitch_chunks(self, chunked_outputs, iou_threshold):
        """Stitch chunks with consistent global IDs."""
        if len(chunked_outputs) <= 1:
            if len(chunked_outputs) == 0:
                return []
            # Single chunk: map all object IDs
            id_mapping = {}
            next_global_id = 0
            for frame_out in chunked_outputs[0]['frames'].values():
                for obj_id in frame_out['object_ids'].tolist():
                    if obj_id not in id_mapping:
                        id_mapping[obj_id] = next_global_id
                        next_global_id += 1
            return [id_mapping]

        id_mappings = [{} for _ in chunked_outputs]
        next_global_id = 0

        # First chunk: map ALL object IDs from all frames
        for frame_out in chunked_outputs[0]['frames'].values():
            for obj_id in frame_out['object_ids'].tolist():
                if obj_id not in id_mappings[0]:
                    id_mappings[0][obj_id] = next_global_id
                    next_global_id += 1

        # Match subsequent chunks
        for i in range(1, len(chunked_outputs)):
            prev_chunk = chunked_outputs[i - 1]
            curr_chunk = chunked_outputs[i]

            prev_last_idx = max(prev_chunk['frames'].keys())

            # Find first frame with detections
            curr_first_idx = None
            for f in sorted(curr_chunk['frames'].keys()):
                if len(curr_chunk['frames'][f]['object_ids']) > 0:
                    curr_first_idx = f
                    break

            if curr_first_idx is None:
                continue

            prev_out = prev_chunk['frames'][prev_last_idx]
            curr_out = curr_chunk['frames'][curr_first_idx]

            prev_masks = prev_out['masks']
            curr_masks = curr_out['masks']
            prev_ids = prev_out['object_ids'].tolist()
            curr_ids = curr_out['object_ids'].tolist()

            # Match current chunk's first frame objects to prev chunk's last frame
            for ci, curr_id in enumerate(curr_ids):
                best_iou, best_prev_id = 0, None
                for pi, prev_id in enumerate(prev_ids):
                    iou = self._compute_mask_iou(prev_masks[pi], curr_masks[ci])
                    if iou > best_iou:
                        best_iou, best_prev_id = iou, prev_id

                if best_iou >= iou_threshold and best_prev_id in id_mappings[i-1]:
                    id_mappings[i][curr_id] = id_mappings[i-1][best_prev_id]
                else:
                    id_mappings[i][curr_id] = next_global_id
                    next_global_id += 1

            # Map remaining object IDs from other frames in this chunk
            for frame_out in curr_chunk['frames'].values():
                for obj_id in frame_out['object_ids'].tolist():
                    if obj_id not in id_mappings[i]:
                        id_mappings[i][obj_id] = next_global_id
                        next_global_id += 1

        return id_mappings

    @staticmethod
    def _build_unified_output(chunked_outputs, id_mappings):
        """
        Build unified output with consistent global IDs across all frames.
        Returns: {global_frame_idx: {global_obj_id: mask, ...}, ...}
        """
        # First pass: find all global IDs that ever appear
        all_global_ids = set()
        for mapping in id_mappings:
            all_global_ids.update(mapping.values())

        unified = {}

        for chunk_idx, chunk in enumerate(chunked_outputs):
            mapping = id_mappings[chunk_idx]
            start_frame = chunk['start_frame']

            for local_frame_idx, frame_out in chunk['frames'].items():
                global_frame_idx = start_frame + local_frame_idx

                # Skip if we already have this frame (from overlap)
                if global_frame_idx in unified:
                    continue

                # Init with None for all known objects
                unified[global_frame_idx] = {gid: None for gid in all_global_ids}

                # Fill in detected masks
                obj_ids = frame_out['object_ids'].tolist()
                masks = frame_out['masks']

                for i, local_id in enumerate(obj_ids):
                    if local_id in mapping:
                        global_id = mapping[local_id]
                        unified[global_frame_idx][global_id] = masks[i]

        # Sort by frame index
        unified = dict(sorted(unified.items()))

        return unified

    def _write_masks(self, unified, mask_dir, frame_rate, frame_size):
        """Write masks to separate video files for each object."""
        if not unified:
            print("No masks to write.")
            return

        # Get all object IDs
        all_obj_ids = set()
        for frame_data in unified.values():
            all_obj_ids.update(frame_data.keys())

        # Create a video writer for each object
        writers = {}
        for obj_id in all_obj_ids:
            writers[obj_id] = VideoWriter(
                path=mask_dir / f"{obj_id}.mp4",
                input_dict={"-r": frame_rate},
                output_dict={"-c:v": "libx264", "-crf": "0", "-pix_fmt": "yuv420p"},
                overwrite=self.overwrite,
            )

        # Write frames
        width, height = frame_size
        for frame_idx in sorted(unified.keys()):
            frame_data = unified[frame_idx]

            for obj_id in all_obj_ids:
                data_tens = frame_data.get(obj_id)
                
                mask = data_tens.detach().cpu().numpy()

                if mask is not None:
                    # Convert boolean mask to RGB
                    mask_rgb = np.repeat((mask * 255).astype(np.uint8)[..., np.newaxis], 3, axis=-1)
                else:
                    # Empty mask
                    mask_rgb = np.zeros((height, width, 3), dtype=np.uint8)

                writers[obj_id].write(image=mask_rgb)

        # Close all writers
        for writer in writers.values():
            writer.close()
