"""pose estimation tool."""

from typing import Union

import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import numpy as np

from psifx.video.tool import VideoTool
from psifx.utils import draw
from psifx.io import tar, video


class PoseEstimationTool(VideoTool):
    """
    Base class for pose estimation tools from video.
    """

    def inference(
        self,
        video_path: Union[str, Path],
        poses_path: Union[str, Path],
    ):
        """
        Template of the inference method.

        :param video_path: Path to the video file.
        :param poses_path: Path to the pose archive.
        :return:
        """
        video_path = Path(video_path)
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
        visualization_path: Union[str, Path],
        confidence_threshold: float = 0.0,
    ):
        """
        Renders a visualization where the estimated poses are overlaid on top of the video.

        :param video_path: Path to the video file.
        :param poses_path: Path to the pose archive.
        :param visualization_path: Path to the visualization file.
        :param confidence_threshold: Threshold for not displaying low confidence keypoints.
        :return:
        """
        assert 0.0 <= confidence_threshold <= 1.0

        video_path = Path(video_path)
        poses_path = Path(poses_path)
        visualization_path = Path(visualization_path)

        if self.verbose:
            print(f"video           =   {video_path}")
            print(f"poses           =   {poses_path}")
            print(f"visualization   =   {visualization_path}")

        assert video_path != visualization_path
        tar.TarReader.check(path=poses_path)

        poses = tar.TarReader.read(
            poses_path,
            verbose=self.verbose,
        )

        try:
            edges = poses.pop("edges.json")
            edges = json.loads(edges)
            edges = {k: tuple(v) for k, v in edges.items()}
        except KeyError:
            print("Missing or incorrect edges.json, only the landmarks will be drawn.")
            pose = next(iter(poses.values()))
            pose = json.loads(pose)
            edges = {key: () for key, value in pose.items()}

        poses = {
            int(k.replace(".json", "")): json.loads(v)
            for k, v in tqdm(
                poses.items(),
                desc="Decoding",
                disable=not self.verbose,
            )
        }

        with (
            video.VideoReader(path=video_path) as video_reader,
            video.VideoWriter(
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
            for image, pose in zip(
                tqdm(
                    video_reader,
                    desc="Processing",
                    disable=not self.verbose,
                ),
                poses.values(),
            ):
                image = Image.fromarray(image.copy())
                for key, value in pose.items():
                    value = np.array(value).reshape(-1, 3)
                    points = value[..., :-1]
                    confidences = value[..., -1:]
                    image = draw.draw_pose(
                        image=image,
                        points=points,
                        edges=edges[key],
                        confidences=confidences >= confidence_threshold,
                        circle_radius=3 if "face" not in key else 0,
                        circle_thickness=1 if "face" not in key else 0,
                        line_thickness=3 if "face" not in key else 1,
                    )
                    # Single color for face? Ellipse with image relative thickness?
                image = np.asarray(image)
                visualization_writer.write(image=image)
