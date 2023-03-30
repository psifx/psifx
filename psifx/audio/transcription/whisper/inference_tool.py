from typing import Union, Dict, Optional

from pathlib import Path
import json

import torch
import torchaudio

from whisper import Whisper, load_model

from psifx.audio.transcription.inference_tool import BaseTranscriptionTool
from psifx.utils.text_writer import (
    TXTWriter,
    VTTWriter,
    SRTWriter,
    TSVWriter,
    JSONWriter,
)


WRITERS = {
    w.suffix: w for w in [TXTWriter, VTTWriter, SRTWriter, TSVWriter, JSONWriter]
}


class WhisperTranscriptionTool(BaseTranscriptionTool):
    def __init__(
        self,
        model_name: str = "small",
        transcription_suffix: str = ".vtt",
        device: str = "cpu",
        overwrite: bool = False,
        verbose: Union[bool, int] = True,
    ):
        super().__init__(
            device=device,
            overwrite=overwrite,
            verbose=verbose,
        )

        self.model_name = model_name
        self.model: Whisper = load_model(model_name, device=self.device)
        # Freeze the model.
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.writer = WRITERS[transcription_suffix]()

    def __call__(
        self,
        audio_path: Union[str, Path],
        transcription_path: Union[str, Path],
        language: Optional[str] = None,
    ):
        if not isinstance(audio_path, Path):
            audio_path = Path(audio_path)
        if not isinstance(transcription_path, Path):
            transcription_path = Path(transcription_path)

        if self.verbose:
            print(f"audio           =   {audio_path}")
            print(f"transcription   =   {transcription_path}")

        # PRE-PROCESSING
        # Nothing to do here, the model wants the path of the audio.

        # INFERENCE
        transcription_results: Dict = self.model.transcribe(
            audio=str(audio_path),
            task="transcribe",
            language=language,
            verbose=self.verbose > 1,
        )

        # POST-PROCESSING
        # Nothing to do here, it is already formatted.

        if transcription_path.exists():
            if self.overwrite:
                transcription_path.unlink()
            else:
                raise FileExistsError(transcription_path)
        transcription_path.parent.mkdir(parents=True, exist_ok=True)
        self.writer(
            result=transcription_results,
            path=transcription_path,
        )


class WhisperTranscriptionWithDiarizationTool(WhisperTranscriptionTool):
    def __call__(
        self,
        audio_path: Union[str, Path],
        diarization_path: Union[str, Path],
        speaker_assignment_path: Union[str, Path],
        transcription_path: Union[str, Path],
        language: Optional[str] = None,
    ):
        if not isinstance(audio_path, Path):
            audio_path = Path(audio_path)
        if not isinstance(diarization_path, Path):
            diarization_path = Path(diarization_path)
        if not isinstance(speaker_assignment_path, Path):
            speaker_assignment_path = Path(speaker_assignment_path)
        if not isinstance(transcription_path, Path):
            transcription_path = Path(transcription_path)

        if self.verbose:
            print(f"audio               =   {audio_path}")
            print(f"diarization         =   {diarization_path}")
            print(f"speaker_assignment  =   {speaker_assignment_path}")
            print(f"transcription       =   {transcription_path}")

        # PRE-PROCESSING
        # We open the audio file, the speaker assignment file to retrieve the identity of the main speaker.
        # We parse the diarization file to retain only the main speaker segments.
        # We only retain those segments in the audio and silence the rest.
        waveform, sample_rate = torchaudio.load(audio_path)
        n_channels, n_frames = waveform.shape

        assert n_channels == 1
        waveform = waveform.flatten()

        with speaker_assignment_path.open() as file:
            speaker_assignment = json.load(file)

        main_speaker = speaker_assignment["main"]

        # Read and parse the segments from file.
        segments = []
        with diarization_path.open() as file:
            for line in file.readlines():
                line = line.split(" ")
                time_start = float(line[3])
                frame_start = int(sample_rate * time_start)
                time_duration = float(line[4])
                frame_duration = int(sample_rate * time_duration)
                frame_end = frame_start + frame_duration
                time_end = float(frame_end / sample_rate)
                speaker = line[7]
                segment = {
                    "time_start": time_start,
                    "time_end": time_end,
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                    "speaker": speaker,
                }
                if speaker == main_speaker:
                    segments.append(segment)

        speaker_mask = torch.full_like(waveform, fill_value=False, dtype=torch.bool)
        for segment in segments:
            speaker_mask[segment["frame_start"] : segment["frame_end"]] = True

        waveform[~speaker_mask] = 0.0

        transcription_results: Dict = self.model.transcribe(
            audio=waveform,
            task="transcribe",
            language=language,
            verbose=self.verbose > 1,
        )

        # POST-PROCESSING
        for segment in transcription_results["segments"]:
            segment["text"] = f"<v {audio_path.stem}>" + segment["text"].strip()

        if transcription_path.exists():
            if self.overwrite:
                transcription_path.unlink()
            else:
                raise FileExistsError(transcription_path)
        transcription_path.parent.mkdir(parents=True, exist_ok=True)
        self.writer(
            result=transcription_results,
            path=transcription_path,
        )


def cli_main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="Path to the input audio or directory containing the input audios.",
    )
    parser.add_argument(
        "--diarization",
        type=Path,
        default=None,
        help="Path to the input diarization or directory containing the input diarizations.",
    )
    parser.add_argument(
        "--speaker_assignment",
        type=Path,
        default=None,
        help="Path to the input speaker assignment or directory containing the input speaker assignments.",
    )
    parser.add_argument(
        "--transcription",
        type=Path,
        required=True,
        help="Path to the output transcription file or directory containing the transcriptions.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="VTT",
        help="Format of the transcription. Available formats: TXT, VTT, SRT, TSV, JSON",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language of the audio, if ignore, the model will try to guess it, it is advised to specify it. "
        "If a directory is passed as input, the language will be the same for all the audio tracks.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="small",
        help="Name of the model, check https://github.com/openai/whisper#available-models-and-languages.",
    )
    # parser.add_argument(
    #     "--add-prefix",
    #     default=False,
    #     action=argparse.BooleanOptionalAction,
    #     help="Adds a prefix relating to the model used.",
    # )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device on which to run the inference, either 'cpu' or 'cuda'.",
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

    if args.diarization is None and args.speaker_assignment is None:
        tool = WhisperTranscriptionTool(
            model_name=args.model_name,
            transcription_suffix=f".{args.format.lower()}",
            device=args.device,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
        if args.audio.is_file():
            audio_path = args.audio
            transcription_path = args.transcription
            tool(
                audio_path=audio_path,
                transcription_path=transcription_path,
                language=args.language,
            )
        elif args.audio.is_dir():
            audio_dir = args.audio
            transcription_dir = args.transcription
            for audio_path in sorted(audio_dir.glob("*.wav")):
                transcription_name = audio_path.stem + tool.writer.suffix
                transcription_path = transcription_dir / transcription_name
                tool(
                    audio_path=audio_path,
                    transcription_path=transcription_path,
                    language=args.language,
                )
        else:
            raise ValueError("args.audio is neither a file or a directory.")

        del tool
    else:
        tool = WhisperTranscriptionWithDiarizationTool(
            model_name=args.model_name,
            transcription_suffix=f".{args.format.lower()}",
            device=args.device,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
        if args.audio.is_file():
            audio_path = args.audio
            diarization_path = args.diarization
            speaker_assignment_path = args.speaker_assignment
            transcription_path = args.transcription
            tool(
                audio_path=audio_path,
                transcription_path=transcription_path,
                diarization_path=diarization_path,
                speaker_assignment_path=speaker_assignment_path,
                language=args.language,
            )
        elif args.audio.is_dir():
            audio_dir = args.audio
            diarization_dir = args.diarization
            speaker_assignment_dir = args.speaker_assignment
            transcription_dir = args.transcription
            for audio_path in sorted(audio_dir.glob("*.wav")):
                transcription_name = audio_path.stem + tool.writer.suffix
                transcription_path = transcription_dir / transcription_name
                diarization_name = audio_path.stem + ".rttm"
                diarization_path = diarization_dir / diarization_name
                speaker_assignment_name = audio_path.stem + ".json"
                speaker_assignment_path = (
                    speaker_assignment_dir / speaker_assignment_name
                )
                tool(
                    audio_path=audio_path,
                    diarization_path=diarization_path,
                    speaker_assignment_path=speaker_assignment_path,
                    transcription_path=transcription_path,
                    language=args.language,
                )
        else:
            raise ValueError("args.audio is neither a file or a directory.")

        del tool
