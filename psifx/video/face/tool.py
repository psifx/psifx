"""face analysis tool."""

from typing import Union

from pathlib import Path

from psifx.video.tool import VideoTool


class FaceAnalysisTool(VideoTool):
    """
    Base tool for face analysis.
    """

    def inference(
        self,
        video_path: Union[str, Path],
        features_path: Union[str, Path],
    ):
        """
        Template of the inference method.

        :param video_path: Path to the video file.
        :param features_path: Path to the feature archive.
        :return:
        """
        video_path = Path(video_path)
        features_path = Path(features_path)

        # video = load(video_path)
        # video = pre_process_func(video)
        # features = model(video)
        # features = post_process_func(features)
        # features.update({"metadata": metastuff})
        # write(features, features_path)

        raise NotImplementedError

    def visualization(
        self,
        video_path: Union[str, Path],
        features_path: Union[str, Path],
        visualization_path: Union[str, Path],
    ):
        """
        Template of the visualization method.

        :param video_path: Path to the video file.
        :param features_path: Path to the feature archive.
        :param visualization_path: Path to the visualization file.
        :return:
        """
        raise NotImplementedError
