"""WhisperX transcription tool."""
from typing import Union, Optional
from pathlib import Path
from psifx.audio.transcription.tool import TranscriptionTool
from psifx.io import vtt, wav
import whisperx


class WhisperXTool(TranscriptionTool):
    """
    WhisperX transcription and translation tool.

    :param model_name: The name of the model to use.
    :param device: The device where the computation should be executed.
    :param overwrite: Whether to overwrite existing files, otherwise raise an error.
    :param verbose: Whether to execute the computation verbosely.
    """

    def __init__(
            self,
            model_name: str = "distil-large-v3",
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
        if task not in ["transcribe", "translate"]:
            raise NameError(f"task should be 'transcribe' or 'translate', got '{task}' instead")
        self.task = task
        compute_type = "float16" if device == 'cuda' else "float32"

        self.pipeline = whisperx.load_model(model_name, task=task, device=device, compute_type=compute_type)

    def inference(
            self,
            audio_path: Union[str, Path],
            transcription_path: Union[str, Path],
            batch_size: int = 16,
            language: Optional[str] = None,
    ):
        """
        WhisperX's backed transcription method.

        :param audio_path: Path to the audio track.
        :param transcription_path: Path to the transcription file.
        :param batch_size: Batch size, reduce if low on GPU memory.
        :param language: Country-code string of the spoken language.
        :return:
        """

        audio_path = Path(audio_path)
        transcription_path = Path(transcription_path)

        if self.verbose:
            print(f"WhisperX")
            print(f"model name      =   {self.model_name}")
            print(f"task            =   {self.task}")
            if language is not None:
                print(f"language        =   {language}")
            print(f"audio           =   {audio_path}")
            print(f"transcription   =   {transcription_path}")

        wav.WAVReader.check(path=audio_path)
        vtt.VTTWriter.check(path=transcription_path, overwrite=self.overwrite)

        audio = whisperx.load_audio(audio_path)
        result = self.pipeline.transcribe(audio, batch_size=batch_size)

        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)

        vtt.VTTWriter.write(
            segments=result["segments"], path=transcription_path, overwrite=self.overwrite
        )
