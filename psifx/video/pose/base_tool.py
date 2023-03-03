from typing import Union

from pathlib import Path

from psifx.base_tool import BaseTool


class BasePoseEstimationTool(BaseTool):
    def __call__(
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
        # write(poses, poses_path)

        raise NotImplementedError
