from typing import Union

from pathlib import Path
from tqdm import tqdm
import json

import numpy as np
from skvideo.io import vreader, ffprobe, FFmpegWriter

from psifx.tool import BaseTool
from psifx.utils import tar, plot


class PoseEstimationTool(BaseTool):
    def inference(
        self,
        video_path: Union[str, Path],
        poses_path: Union[str, Path],
    ):
        if not isinstance(video_path, Path):
            video_path = Path(video_path)
        if not isinstance(poses_path, Path):
            poses_path = Path(poses_path)

        # video = load(video_path)
        # video = pre_process_func(video)
        # poses = model(video)
        # poses = post_process_func(poses)
        # poses.update({"edges": self.edges})
        # write(poses, poses_path)

        raise NotImplementedError

    def visualization(
        self,
        video_path: Union[str, Path],
        poses_path: Union[str, Path],
        visualisation_path: Union[str, Path],
        confidence_threshold: float = 0.0,
    ):
        if not isinstance(video_path, Path):
            video_path = Path(video_path)
        if not isinstance(poses_path, Path):
            poses_path = Path(poses_path)
        if not isinstance(visualisation_path, Path):
            visualisation_path = Path(visualisation_path)

        if self.verbose:
            print(f"video           =   {video_path}")
            print(f"poses           =   {poses_path}")
            print(f"visualisation   =   {visualisation_path}")

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
            pose = next(iter(dictionary.values()))
            edges = {key: () for key, value in pose.items()}

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
                image = image.copy()

                for key, value in pose.items():
                    value = np.array(value).reshape(-1, 3)
                    points = value[..., :-1]
                    confidences = value[..., -1:]
                    image = plot.draw_pose(
                        image=image,
                        points=points,
                        confidences=confidences >= confidence_threshold,
                        edges=edges[key],
                        draw_points="face" not in key,
                    )

                visualisation_writer.writeFrame(image)


def visualization_main():
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

    tool = PoseEstimationTool(
        device="cpu",
        overwrite=args.overwrite,
        verbose=args.verbose,
    )
    tool.visualization(
        video_path=args.video,
        poses_path=args.poses,
        visualisation_path=args.visualisation,
        confidence_threshold=args.confidence_threshold,
    )
    del tool
