from typing import Union, Optional

from pathlib import Path

import torch
from whisper import Whisper, load_model

from psifx.audio.transcription.tool import TranscriptionTool
from psifx.io import vtt, wav


class WhisperTranscriptionTool(TranscriptionTool):
    """
    Whisper transcription and translation tool.
    """

    def __init__(
        self,
        model_name: str = "small",
        task: str = "transcribe",
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
        self.task = task
        self.model: Whisper = load_model(model_name, device=self.device)
        # Freeze the model.
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def inference(
        self,
        audio_path: Union[str, Path],
        transcription_path: Union[str, Path],
        language: Optional[str] = None,
    ):
        """
        Whisper's backed transcription method.

        :param audio_path: Path to the audio track.
        :param transcription_path: Path to the transcription file.
        :param language: Country-code string of the spoken language.
        :return:
        """
        audio_path = Path(audio_path)
        transcription_path = Path(transcription_path)

        if self.verbose:
            print(f"audio           =   {audio_path}")
            print(f"transcription   =   {transcription_path}")

        wav.WAVReader.check(audio_path)
        vtt.VTTWriter.check(transcription_path)

        # PRE-PROCESSING
        # Nothing to do here, the model wants the path of the audio.

        # INFERENCE
        with torch.no_grad():
            segments = self.model.transcribe(
                audio=str(audio_path),
                task=self.task,
                language=language,
                verbose=self.verbose > 1,
            )["segments"]

        # POST-PROCESSING
        vtt.VTTWriter.write(
            segments=segments, path=transcription_path, overwrite=self.overwrite
        )
