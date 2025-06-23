"""samurai tracking tool."""

import numpy as np
import torch

from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.build_sam import build_sam2_video_predictor
from urllib.request import urlretrieve

from ultralytics import YOLO
from ultralytics.engine.results import Results

from typing import Union, Optional
from pathlib import Path

from psifx.io.video import VideoReader, VideoWriter
from psifx.video.tracking.tool import TrackingTool

from platformdirs import user_cache_dir

# Base URL for SAM 2.1 checkpoints
SAM2p1_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"

# Available models and their corresponding filenames
CHECKPOINT_FILES = {
    "tiny": "sam2.1_hiera_tiny.pt",
    "small": "sam2.1_hiera_small.pt",
    "base_plus": "sam2.1_hiera_base_plus.pt",
    "large": "sam2.1_hiera_large.pt"
}
CONFIG_FILES = {
    "tiny": "sam2.1_hiera_t.yaml",
    "small": "sam2.1_hiera_s.yaml",
    "base_plus": "sam2.1_hiera_b+.yaml",
    "large": "sam2.1_hiera_l.yaml"
}


def download_checkpoint(model_size: str):
    if model_size not in CHECKPOINT_FILES:
        raise ValueError(f"Invalid model size '{model_size}'. Choose from: {', '.join(CHECKPOINT_FILES.keys())}")
    filename = CHECKPOINT_FILES[model_size]

    cache_dir = Path(user_cache_dir("psifx")) / "sam-2"
    cache_dir.mkdir(parents=True, exist_ok=True)
    file_path = cache_dir / filename
    if not file_path.exists():
        url = f"{SAM2p1_BASE_URL}/{filename}"
        print(f"Downloading {filename} from {url}...")
        try:
            urlretrieve(url, file_path)
            print(f"Downloaded and saved to: {file_path}")
        except Exception as e:
            print(f"Failed to download checkpoint: {e}")

    return file_path


class SamuraiTrackingTool(TrackingTool):
    def __init__(self,
                 model_size: str = "tiny",
                 use_samurai: bool = True,
                 yolo_model: str = "yolo11n.pt",
                 device: str = "cpu",
                 overwrite: bool = False,
                 verbose: Union[bool, int] = True,
                 ):
        super().__init__(
            device=device,
            overwrite=overwrite,
            verbose=verbose,
        )

        self.sam2_checkpoint = download_checkpoint(model_size=model_size)
        self.sam2_cfg = f"configs/{'samurai' if use_samurai else 'sam2.1'}/{CONFIG_FILES[model_size]}"
        cache_dir = Path(user_cache_dir("psifx"))
        self.yolo_model = YOLO(cache_dir / "ultralytics" / yolo_model)

        if device == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif device == "mps":
            print(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS. "
                "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )

        self.predictor: SAM2VideoPredictor = build_sam2_video_predictor(self.sam2_cfg, self.sam2_checkpoint,
                                                                        device=self.device)

    def infer(self,
              video_path: Union[str, Path],
              mask_dir: Union[str, Path],
              object_class: Optional[int] = 0,
              max_objects: Optional[int] = None,
              step_size: Optional[int] = 30):
        """
        Perform object detection, segmentation, and tracking from a video file.

        :param video_path: Path to the input video.
        :param mask_dir: Path to the output mask directory.
        :param object_class: Class of the object to detect according to Yolo (0 is a person).
        :param max_objects: Max number of objects to detect. If None, go through the whole video.
        :param step_size: Frame interval to perform detection.
        """
        mask_dir = Path(mask_dir)
        if mask_dir.exists() and any(mask_dir.iterdir()):
            if self.overwrite:
                print(f"Mask directory {mask_dir} is non-empty")
            else:
                raise FileExistsError(f"Mask directory {mask_dir} is non-empty.")

        identified_objects = []
        frame_idx = 0
        detected_frame = None

        if step_size <= 0:
            raise ValueError(f"step_size should be a positive integer, not {step_size}")
        if not (0 <= object_class <= 79):
            raise ValueError(f"object_class should be between 0 and 79 inclusive, not {object_class}")

        with (VideoReader(path=video_path) as video_reader):
            frame_rate = video_reader.frame_rate
            for image in video_reader:
                if frame_idx % step_size == 0:
                    result: Results = self.yolo_model(image, verbose=False)[0]

                    object_detected = [list(summary['box'].values()) for summary in result.summary() if
                                       summary['class'] == object_class]

                    if len(object_detected) > len(identified_objects):
                        identified_objects = object_detected
                        detected_frame = frame_idx
                        if max_objects is not None and len(identified_objects) >= max_objects:
                            break
                frame_idx += 1

        if len(identified_objects) == 0:
            raise RuntimeError(f"Failed to detect any object of class {object_class} in the video.")

        for object_id, object_box in enumerate(identified_objects, start=1):

            inference_state = self.predictor.init_state(video_path=str(video_path))

            box = np.array(object_box, dtype=np.float32)

            self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=detected_frame,
                obj_id=object_id,
                box=box,
            )

            object_masks = []

            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                    start_frame_idx=0,
                    inference_state=inference_state
            ):
                binary_mask = (out_mask_logits.squeeze() > 0.0).cpu().numpy().astype(np.uint8)
                mask_data = binary_mask * 255
                # Convert to RGB by stacking
                mask_rgb = np.repeat(mask_data[..., np.newaxis], 3, axis=-1)
                object_masks.append(mask_rgb)

            with VideoWriter(
                    path=mask_dir / f"{object_id}.mp4",
                    input_dict={"-r": frame_rate},
                    output_dict={"-c:v": "libx264", "-crf": "0", "-pix_fmt": "yuv420p"},
                    overwrite=self.overwrite,
            ) as mask_writer:
                for mask in object_masks:
                    mask_writer.write(image=mask)
