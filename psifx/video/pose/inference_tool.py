from typing import Dict, Tuple, Union

from pathlib import Path

from psifx.base_tool import BaseTool


class BasePoseEstimationTool(BaseTool):
    def __init__(
        self,
        edges: Dict[str, Tuple[Tuple[int, int], ...]],
        device: str = "cpu",
        overwrite: bool = False,
        verbose: Union[bool, int] = True,
    ):
        super().__init__(
            device=device,
            overwrite=overwrite,
            verbose=verbose,
        )
        self.edges = edges

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
        # poses.update({"edges": self.edges})
        # write(poses, poses_path)

        raise NotImplementedError
