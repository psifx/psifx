from typing import Sequence, Union

from pathlib import Path
import ffmpeg

from pydub import AudioSegment

from psifx.tool import BaseTool


class ManipulationTool(BaseTool):
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

    def extraction(
        self,
        video_path: Union[str, Path],
        audio_path: Union[str, Path],
    ):
        video_path = Path(video_path)
        audio_path = Path(audio_path)

        if self.verbose:
            print(f"video   =   {video_path}")
            print(f"audio   =   {audio_path}")

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

    def convert(
        self,
        audio_path: Union[str, Path],
        mono_audio_path: Union[str, Path],
    ):
        audio_path = Path(audio_path)
        mono_audio_path = Path(mono_audio_path)

        assert mono_audio_path.suffix == ".wav"

        if self.verbose:
            print(f"audio           =   {audio_path}")
            print(f"converted_audio =   {mono_audio_path}")

        mono_audios = [
            audio.apply_gain(-audio.max_dBFS - 6.0)
            for audio in AudioSegment.from_file(audio_path).split_to_mono()
        ]
        mono_audio = mono_audios[0]
        for audio in mono_audios:
            mono_audio.overlay(audio)

        if mono_audio_path.exists():
            if self.overwrite:
                mono_audio_path.unlink()
            else:
                raise FileExistsError(mono_audio_path)
        mono_audio_path.parent.mkdir(parents=True, exist_ok=True)

        mono_audio.export(mono_audio_path, format="wav")

    def mixdown(
        self,
        mono_audio_paths: Sequence[Union[str, Path]],
        mixed_audio_path: Union[str, Path],
    ):
        mono_audio_paths = [Path(path) for path in mono_audio_paths]
        mixed_audio_path = Path(mixed_audio_path)

        if self.verbose:
            print(f"mono_audios     = {[str(path) for path in mono_audio_paths]}")
            print(f"mixed_audio     = {mixed_audio_path}")

        mono_audios = [AudioSegment.from_wav(path) for path in mono_audio_paths]
        mono_audios = [audio.apply_gain(-audio.max_dBFS - 6.0) for audio in mono_audios]
        mixed_audio = mono_audios[0]
        for audio in mono_audios[1:]:
            mixed_audio.overlay(audio)

        if mixed_audio_path.exists():
            if self.overwrite:
                mixed_audio_path.unlink()
            else:
                raise FileExistsError(mixed_audio_path)
        mixed_audio_path.parent.mkdir(parents=True, exist_ok=True)

        mixed_audio.export(mixed_audio_path, format="wav")

    def normalization(
        self,
        audio_path: Union[str, Path],
        normalized_audio_path: Union[str, Path],
    ):
        audio_path = Path(audio_path)
        normalized_audio_path = Path(normalized_audio_path)

        if self.verbose:
            print(f"audio               =   {audio_path}")
            print(f"normalized_audio    =   {normalized_audio_path}")

        audio = AudioSegment.from_wav(audio_path)
        normalized_audio = audio.apply_gain(-audio.max_dBFS)

        if normalized_audio_path.exists():
            if self.overwrite:
                normalized_audio_path.unlink()
            else:
                raise FileExistsError(normalized_audio_path)
        normalized_audio_path.parent.mkdir(parents=True, exist_ok=True)
        normalized_audio.export(normalized_audio_path, format="wav")


def extraction_main():
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

    tool = ManipulationTool(
        overwrite=args.overwrite,
        verbose=args.verbose,
    )
    tool.extraction(
        video_path=args.video,
        audio_path=args.audio,
    )
    del tool


def mixdown_main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mono_audios",
        nargs="+",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--mixed_audio",
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

    tool = ManipulationTool(
        overwrite=args.overwrite,
        verbose=args.verbose,
    )
    tool.mixdown(
        mono_audio_paths=args.mono_audios,
        mixed_audio_path=args.mixed_audio,
    )
    del tool


def normalization_main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--normalized_audio",
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

    tool = ManipulationTool(
        overwrite=args.overwrite,
        verbose=args.verbose,
    )
    tool.normalization(
        audio_path=args.audio,
        normalized_audio_path=args.normalized_audio,
    )
    del tool
