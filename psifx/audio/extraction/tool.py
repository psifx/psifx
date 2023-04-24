from typing import Union

from pathlib import Path
import ffmpeg

from psifx.tool import BaseTool


class ExtractionTool(BaseTool):
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

    def inference(
        self,
        in_video_path: Union[str, Path],
        out_audio_path: Union[str, Path],
    ):
        if not isinstance(in_video_path, Path):
            in_video_path = Path(in_video_path)
        if not isinstance(out_audio_path, Path):
            out_audio_path = Path(out_audio_path)

        if self.verbose:
            print(f"in_video     =   {in_video_path}")
            print(f"out_audio    =   {out_audio_path}")

        if out_audio_path.exists():
            if self.overwrite:
                out_audio_path.unlink()
            else:
                raise FileExistsError(out_audio_path)
        out_audio_path.parent.mkdir(parents=True, exist_ok=True)
        (
            ffmpeg.input(str(in_video_path))
            .audio.output(str(out_audio_path), **{"q:a": 0, "ac": 1, "ar": 16000})
            .overwrite_output()
            .run(quiet=not self.verbose > 1)
        )


def inference_main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_video",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--out_audio",
        type=Path,
        required=True,
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

    tool = ExtractionTool(
        overwrite=args.overwrite,
        verbose=args.verbose,
    )
    tool.inference(
        in_video_path=args.in_video,
        out_audio_path=args.out_audio,
    )
    del tool
