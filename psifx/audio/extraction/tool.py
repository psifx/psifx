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
        video_path: Union[str, Path],
        audio_path: Union[str, Path],
    ):
        video_path = Path(video_path)
        audio_path = Path(audio_path)

        if self.verbose:
            print(f"in_video     =   {video_path}")
            print(f"out_audio    =   {audio_path}")

        if audio_path.exists():
            if self.overwrite:
                audio_path.unlink()
            else:
                raise FileExistsError(audio_path)
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        (
            ffmpeg.input(str(video_path))
            .audio.output(str(audio_path), **{"q:a": 0, "ac": 1, "ar": 16000})
            .overwrite_output()
            .run(quiet=not self.verbose > 1)
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
        "--audio",
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
        video_path=args.video,
        audio_path=args.audio,
    )
    del tool
