"""tracking tool."""

import numpy as np
import random
import cv2

from typing import Union
from pathlib import Path
from tqdm import tqdm

from psifx.io.video import VideoReader, VideoWriter
from psifx.video.tool import VideoTool


class TrackingTool(VideoTool):
    def __init__(self,
                 device: str,
                 overwrite: bool = False,
                 verbose: Union[bool, int] = True,
                 ):
        super().__init__(
            device=device,
            overwrite=overwrite,
            verbose=verbose,
        )

    def visualize(self,
                  video_path: Union[str, Path],
                  mask_paths: Union[str, Path, list[Union[str, Path]]],
                  visualization_path: Union[str, Path],
                  blackout: bool = False,
                  color: bool = True,
                  labels: bool = True,
                  ):
        """
        Applies color masks to the video and writes a new video.

        :param video_path: Path to original video.
        :param mask_paths: Single or list of paths to .mp4 video masks.
        :param visualization_path: Path to save the output video.
        :param blackout: Whether to black out the background.
        :param color: Whether to color each mask.
        :param labels: Whether to draw labels.
        """
        if not isinstance(mask_paths, list):
            mask_paths = [mask_paths]
        mask_paths = [Path(p) for p in mask_paths]
        for path in mask_paths:
            if not path.suffix == ".mp4":
                raise ValueError(f"Expected .mp4 file, got {path}")
            if not path.exists():
                raise FileNotFoundError(f"File missing at path {path}")

        obj_ids = [str(p.stem) for p in mask_paths]

        if color:
            obj_colors = {
                obj_id: np.array([random.randint(50, 255) for _ in range(3)], dtype=np.uint8)
                for obj_id in obj_ids
            }

        mask_readers = {obj_id: VideoReader(path=path) for obj_id, path in zip(obj_ids, mask_paths)}

        with (
            VideoReader(path=video_path) as video_reader,
            VideoWriter(
                path=visualization_path,
                input_dict={"-r": video_reader.frame_rate},
                output_dict={
                    "-c:v": "libx264",
                    "-crf": "15",
                    "-pix_fmt": "yuv420p",
                },
                overwrite=self.overwrite,
            ) as visualization_writer,
        ):
            for frame_idx, frame in enumerate(tqdm(video_reader, desc="Overlaying masks", disable=not self.verbose)):
                composite_mask = np.zeros_like(frame, dtype=np.uint8)

                if blackout:
                    background = np.zeros_like(frame)
                else:
                    background = frame

                label_positions = {}

                for obj_id, reader in mask_readers.items():
                    try:
                        mask_frame = next(reader)
                    except StopIteration:
                        continue

                    # Convert mask to binary mask
                    gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
                    binary_mask = gray > 127

                    if np.any(binary_mask):

                        if color:
                            obj_color = obj_colors[obj_id]
                            for c in range(3):
                                composite_mask[..., c] += (binary_mask * obj_color[c]).astype(np.uint8)

                        if labels:
                            ys, xs = np.where(binary_mask)
                            if len(xs) > 0 and len(ys) > 0:
                                mean_y = int(np.mean(ys))
                                mean_x = int(np.mean(xs))
                                label_positions[obj_id] = (mean_x, mean_y - 10)

                        if blackout:
                            background[binary_mask] = frame[binary_mask]

                if color:
                    overlay = cv2.addWeighted(background, 1.0, composite_mask, 0.5, 0)
                else:
                    overlay = background

                if labels:
                    self.draw_labels(overlay, label_positions)

                visualization_writer.write(image=overlay)

        for reader in mask_readers.values():
            reader.close()

    @staticmethod
    def draw_labels(overlay, label_positions):
        for key, pos_tuple in label_positions.items():
            x, y = pos_tuple

            # Calculate text size
            (text_width, text_height), baseline = cv2.getTextSize(
                key,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                thickness=1
            )

            # Adjust position to center the text
            x_centered = x - text_width // 2
            y_adjusted = max(y, text_height + baseline)

            cv2.putText(
                overlay,
                key,
                (x_centered, y_adjusted),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 0, 0),  # black
                thickness=3,
                lineType=cv2.LINE_AA,
            )

            cv2.putText(
                overlay,
                key,
                (x_centered, y_adjusted),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 255, 255),  # white
                thickness=1,
                lineType=cv2.LINE_AA,
            )
