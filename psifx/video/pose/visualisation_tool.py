from typing import Union

from pathlib import Path
from tqdm import tqdm
import json

import numpy as np
from skvideo.io import vreader, ffprobe, FFmpegWriter

from psifx.base_tool import BaseTool
from psifx.utils import tar, plot


class VisualisationTool(BaseTool):
    def __init__(
        self,
        confidence_threshold: float = 0.0,
        overwrite: bool = False,
        verbose: Union[bool, int] = True,
    ):
        super().__init__(
            device="cpu",
            overwrite=overwrite,
            verbose=verbose,
        )
        self.confidence_threshold = confidence_threshold

    def __call__(
        self,
        video_path: Union[str, Path],
        poses_path: Union[str, Path],
        visualisation_path: Union[str, Path],
    ):
        if not isinstance(video_path, Path):
            video_path = Path(video_path)
        if not isinstance(poses_path, Path):
            poses_path = Path(poses_path)
        if not isinstance(visualisation_path, Path):
            visualisation_path = Path(visualisation_path)

        assert video_path != visualisation_path

        dictionary = {
            k.replace(".json", ""): json.loads(v)
            for k, v in tar.load(poses_path).items()
        }
        try:
            edges = dictionary.pop("edges")
            edges = {k: tuple(v) for k, v in edges.items()}
        except KeyError:
            print("Missing or incorrect edges.json, only the landmarks will be drawn.")
            edges = {
                "pose_keypoints_edges": (),
                "face_keypoints_edges": (),
                "hand_left_keypoints_edges": (),
                "hand_right_keypoints_edges": (),
            }

        poses = {int(k): v for k, v in dictionary.items()}

        video_info = ffprobe(str(video_path))
        frame_rate = video_info["video"]["@r_frame_rate"]

        if visualisation_path.exists():
            if self.overwrite:
                visualisation_path.unlink()
            else:
                raise FileExistsError(visualisation_path)
        visualisation_path.parent.mkdir(parents=True, exist_ok=True)

        with FFmpegWriter(
            filename=str(visualisation_path),
            inputdict={"-r": frame_rate},
            outputdict={"-c:v": "libx264", "-crf": "15", "-pix_fmt": "yuv420p"},
        ) as visualisation_writer:
            for i, (image, pose) in enumerate(
                zip(
                    vreader(str(video_path)),
                    tqdm(
                        poses.values(),
                        disable=not self.verbose,
                    ),
                )
            ):
                pose = {k: np.array(v).reshape(-1, 3) for k, v in pose.items()}
                image = image.copy()

                points = pose["pose_keypoints_2d"][..., :-1]
                confidences = pose["pose_keypoints_2d"][..., -1:]
                image = plot.draw_pose(
                    image=image,
                    points=points,
                    confidences=confidences >= self.confidence_threshold,
                    edges=edges["pose_keypoints_edges"],
                )

                points = pose["face_keypoints_2d"][..., :-1]
                confidences = pose["face_keypoints_2d"][..., -1:]
                image = plot.draw_pose(
                    image=image,
                    points=points,
                    confidences=confidences >= self.confidence_threshold,
                    edges=edges["face_keypoints_edges"],
                    draw_points=False,
                )

                points = pose["hand_left_keypoints_2d"][..., :-1]
                confidences = pose["hand_left_keypoints_2d"][..., -1:]
                image = plot.draw_pose(
                    image=image,
                    points=points,
                    confidences=confidences >= self.confidence_threshold,
                    edges=edges["hand_left_keypoints_edges"],
                )

                points = pose["hand_right_keypoints_2d"][..., :-1]
                confidences = pose["hand_right_keypoints_2d"][..., -1:]
                image = plot.draw_pose(
                    image=image,
                    points=points,
                    confidences=confidences >= self.confidence_threshold,
                    edges=edges["hand_right_keypoints_edges"],
                )

                visualisation_writer.writeFrame(image)


def cli_main():
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
        "--visualisation",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.0,
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

    tool = VisualisationTool(
        confidence_threshold=args.confidence_threshold,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )

    if args.video.is_file():
        video_path = args.video
        poses_path = args.poses
        visualisation_path = args.visualisation
        tool(
            video_path=video_path,
            poses_path=poses_path,
            visualisation_path=visualisation_path,
        )
    elif args.video.is_dir():
        video_dir = args.video
        poses_dir = args.poses
        visualisation_dir = args.visualisation
        for video_path in sorted(video_dir.glob("*.mp4")):
            poses_name = video_path.stem + ".tar.gz"
            poses_path = poses_dir / poses_name
            visualisation_name = video_path.stem + ".mp4"
            visualisation_path = visualisation_dir / visualisation_name
            tool(
                video_path=video_path,
                poses_path=poses_path,
                visualisation_path=visualisation_path,
            )
    else:
        raise ValueError("args.video is neither a file or a directory.")
