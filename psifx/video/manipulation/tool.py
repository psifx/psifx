from typing import Optional, Union

from pathlib import Path
import ffmpeg

from psifx.video.tool import VideoTool


class ManipulationTool(VideoTool):
    def __init__(
        self,
        overwrite: bool = False,
        verbose: Union[bool, int] = True,
    ):
        super().__init__(
            device="cpu",
            overwrite=overwrite,
            verbose=verbose,
        )

    def process(
        self,
        in_video_path: Union[str, Path],
        out_video_path: Union[str, Path],
        start: Optional[float] = None,
        end: Optional[float] = None,
        x_min: Optional[int] = None,
        y_min: Optional[int] = None,
        x_max: Optional[int] = None,
        y_max: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ):
        in_video_path = Path(in_video_path)
        out_video_path = Path(out_video_path)

        if self.verbose:
            print(f"in_video    =   {in_video_path}")
            print(f"out_video   =   {out_video_path}")

        assert in_video_path != out_video_path

        crop = all(p is not None for p in [x_min, y_min, x_max, y_max])
        no_crop = all(p is None for p in [x_min, y_min, x_max, y_max])
        resize = all(p is not None for p in [width, height])
        no_resize = all(p is None for p in [width, height])

        assert crop or no_crop
        assert resize or no_resize

        kwargs = {}
        if start is not None:
            kwargs.update(ss=start)
        if end is not None:
            kwargs.update(to=end)
        input = ffmpeg.input(str(in_video_path), **kwargs)

        video = input.video
        audio = input.audio
        if crop:
            assert x_min < x_max and y_min < y_max
            video = video.crop(
                x=x_min,
                y=y_min,
                width=x_max - x_min,
                height=y_max - y_min,
            )

        if resize:
            video = video.filter("scale", f"{width}x{height}")

        output = ffmpeg.output(video, audio, str(out_video_path))

        if out_video_path.exists():
            if self.overwrite:
                out_video_path.unlink()
            else:
                raise FileExistsError(out_video_path)
        out_video_path.parent.mkdir(parents=True, exist_ok=True)

        output.overwrite_output().run(quiet=not self.verbose > 1)
