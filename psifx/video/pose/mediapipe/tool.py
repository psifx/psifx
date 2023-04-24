from typing import Any, Dict, List, Tuple, Union

from pathlib import Path
from tqdm import tqdm
import json

import numpy as np
from skvideo.io import vreader, ffprobe, FFmpegWriter
from mediapipe.python.solutions.holistic import Holistic

from psifx.video.pose.tool import PoseEstimationTool, visualization_main
from psifx.video.pose.mediapipe import skeleton
from psifx.utils import tar


class MediaPipePoseEstimationTool(PoseEstimationTool):
    def __init__(
        self,
        model_complexity: int = 2,
        smooth: bool = True,
        device: str = "cpu",
        overwrite: bool = False,
        verbose: Union[bool, int] = True,
    ):
        super().__init__(
            device=device,
            overwrite=overwrite,
            verbose=verbose,
        )

        assert self.device == "cpu", "Only CPU support is currently available."
        self.model_complexity = model_complexity
        self.smooth = smooth

    def process_part(
        self,
        landmarks,
        size: Tuple[int, int],
        n_points: int,
    ) -> List[float]:
        h, w = size
        if landmarks is not None:
            landmarks = [
                [
                    p.x * (w - 1),
                    p.y * (h - 1),
                    p.visibility,
                ]
                for p in landmarks.landmark
            ]
            points = np.array(landmarks, dtype=np.float32)
        else:
            points = np.zeros((n_points, 3), dtype=np.float32)
        return points.flatten().tolist()

    def process_pose(
        self,
        results,
        size: Tuple[int, int],
    ) -> Dict[str, Any]:
        return {
            "pose_keypoints_2d": self.process_part(
                landmarks=results.pose_landmarks,
                size=size,
                n_points=skeleton.N_POSE_LANDMARKS,
            ),
            "face_keypoints_2d": self.process_part(
                landmarks=results.face_landmarks,
                size=size,
                n_points=skeleton.N_FACE_LANDMARKS,
            ),
            "hand_left_keypoints_2d": self.process_part(
                landmarks=results.left_hand_landmarks,
                size=size,
                n_points=skeleton.N_LEFT_HAND_LANDMARKS,
            ),
            "hand_right_keypoints_2d": self.process_part(
                landmarks=results.right_hand_landmarks,
                size=size,
                n_points=skeleton.N_LEFT_HAND_LANDMARKS,
            ),
        }

    def inference(
        self,
        video_path: Union[str, Path],
        poses_path: Union[str, Path],
    ):
        if not isinstance(video_path, Path):
            video_path = Path(video_path)
        if not isinstance(poses_path, Path):
            poses_path = Path(poses_path)

        if self.verbose:
            print(f"video   =   {video_path}")
            print(f"poses   =   {poses_path}")

        video_info = ffprobe(str(video_path))
        n_frames = int(video_info["video"]["@nb_frames"])

        poses = {
            "edges": {
                "pose_keypoints_2d": skeleton.POSE_EDGES,
                "face_keypoints_2d": skeleton.FACE_EDGES,
                "hand_left_keypoints_2d": skeleton.LEFT_HAND_EDGES,
                "hand_right_keypoints_2d": skeleton.RIGHT_HAND_EDGES,
            }
        }

        # We have to instantiate the model for every __call__, because of internal states.
        # Not that it is very costly anyway.
        with Holistic(
            static_image_mode=False,
            model_complexity=self.model_complexity,
            smooth_landmarks=self.smooth,
            enable_segmentation=False,
            smooth_segmentation=False,
            refine_face_landmarks=True,
        ) as model:
            for i, image in enumerate(
                tqdm(
                    vreader(str(video_path)),
                    total=n_frames,
                    disable=not self.verbose,
                )
            ):
                h, w, c = image.shape
                results = model.process(image)
                poses[f"{i: 015d}"] = self.process_pose(
                    results=results,
                    size=(h, w),
                )

        if poses_path.exists():
            if self.overwrite:
                poses_path.unlink()
            else:
                raise FileExistsError(poses_path)
        poses_path.parent.mkdir(parents=True, exist_ok=True)
        tar.dump(
            dictionary={f"{k}.json": json.dumps(v) for k, v in poses.items()},
            path=poses_path,
        )


class MediaPipePoseEstimationAndSegmentationTool(MediaPipePoseEstimationTool):
    def __init__(
        self,
        model_complexity: int = 2,
        smooth: bool = True,
        mask_threshold: float = 0.1,
        device: str = "cpu",
        overwrite: bool = False,
        verbose: Union[bool, int] = True,
    ):
        super().__init__(
            model_complexity=model_complexity,
            smooth=smooth,
            device=device,
            overwrite=overwrite,
            verbose=verbose,
        )
        self.mask_threshold = mask_threshold

    def process_mask(
        self,
        mask: np.ndarray,
        size: Tuple[int, int],
        threshold: float,
    ) -> np.ndarray:
        h, w = size
        if mask is not None:
            mask = np.array(mask)
            mask[mask < threshold] = 0.0
            mask[mask > threshold] = 1.0
            mask = (mask * 255.0).astype(dtype=np.uint8)
        else:
            mask = np.zeros((h, w), dtype=np.uint8)
        return np.stack((mask,) * 3, axis=-1)

    def inference(
        self,
        video_path: Union[str, Path],
        poses_path: Union[str, Path],
        masks_path: Union[str, Path],
    ):
        if not isinstance(video_path, Path):
            video_path = Path(video_path)
        if not isinstance(poses_path, Path):
            poses_path = Path(poses_path)
        if not isinstance(masks_path, Path):
            masks_path = masks_path

        if self.verbose:
            print(f"video   =   {video_path}")
            print(f"poses   =   {poses_path}")
            print(f"masks   =   {masks_path}")

        video_info = ffprobe(str(video_path))
        n_frames = int(video_info["video"]["@nb_frames"])
        frame_rate = video_info["video"]["@r_frame_rate"]

        poses = {
            "edges": {
                "pose_keypoints_2d": skeleton.POSE_EDGES,
                "face_keypoints_2d": skeleton.FACE_EDGES,
                "hand_left_keypoints_2d": skeleton.LEFT_HAND_EDGES,
                "hand_right_keypoints_2d": skeleton.RIGHT_HAND_EDGES,
            }
        }

        if masks_path.exists():
            if self.overwrite:
                masks_path.unlink()
            else:
                raise FileExistsError(masks_path)
        masks_path.parent.mkdir(parents=True, exist_ok=True)

        # We have to instantiate the model for every __call__, because of internal states.
        # Not that it is very costly anyway.
        with (
            Holistic(
                static_image_mode=False,
                model_complexity=self.model_complexity,
                smooth_landmarks=self.smooth,
                enable_segmentation=True,
                smooth_segmentation=self.smooth,
                refine_face_landmarks=True,
            ) as model,
            FFmpegWriter(
                filename=str(masks_path),
                inputdict={"-r": frame_rate},
                outputdict={"-c:v": "libx264", "-crf": "0", "-pix_fmt": "yuv420p"},
            ) as mask_writer,
        ):
            for i, image in enumerate(
                tqdm(
                    vreader(str(video_path)),
                    total=n_frames,
                    disable=not self.verbose,
                )
            ):
                h, w, c = image.shape
                results = model.process(image)
                poses[f"{i: 015d}"] = self.process_pose(
                    results=results,
                    size=(h, w),
                )
                mask = self.process_mask(
                    mask=results.segmentation_mask,
                    size=(h, w),
                    threshold=self.mask_threshold,
                )
                mask_writer.writeFrame(mask)

        if poses_path.exists():
            if self.overwrite:
                poses_path.unlink()
            else:
                raise FileExistsError(poses_path)
        poses_path.parent.mkdir(parents=True, exist_ok=True)
        tar.dump(
            dictionary={f"{k}.json": json.dumps(v) for k, v in poses.items()},
            path=poses_path,
        )


def inference_main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--poses",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--masks",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--mask_threshold",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--model_complexity",
        type=int,
        default=2,
        help="Complexity of the model, either 0, 1 or 2. Higher means more FLOPs, but also more accurate results.",
    )
    parser.add_argument(
        "--smooth",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to temporally smooth the inference results to reduce the jitter.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device on which to run the inference, either 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Overwrite existing files, otherwise raises an error.",
    )
    parser.add_argument(
        "--verbose",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Verbosity of the script.",
    )
    args = parser.parse_args()

    if args.masks is None:
        tool = MediaPipePoseEstimationTool(
            model_complexity=args.model_complexity,
            smooth=args.smooth,
            device=args.device,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
        tool.inference(
            video_path=args.video,
            poses_path=args.poses,
        )
    else:
        tool = MediaPipePoseEstimationAndSegmentationTool(
            model_complexity=args.model_complexity,
            smooth=args.smooth,
            mask_threshold=args.mask_threshold,
            device=args.device,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
        tool.inference(
            video_path=args.video,
            poses_path=args.poses,
            masks_path=args.masks,
        )
    del tool
