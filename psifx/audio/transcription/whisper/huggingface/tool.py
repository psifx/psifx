"""HuggingFace Whisper transcription tool."""
import os
from typing import Union, Optional
from pathlib import Path
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from psifx.audio.transcription.tool import TranscriptionTool
from psifx.io import vtt, wav


class HuggingFaceWhisperTool(TranscriptionTool):
    """
    HuggingFace Whisper transcription and translation tool.

    :param model_name: The name of the model to use.
    :param device: The device where the computation should be executed.
    :param overwrite: Whether to overwrite existing files, otherwise raise an error.
    :param verbose: Whether to execute the computation verbosely.
    """

    def __init__(
            self,
            model_name: str = "openai/whisper-small",
            api_token: Optional[str] = None,
            device: str = "cpu",
            overwrite: bool = False,
            verbose: Union[bool, int] = True,
    ):
        super().__init__(
            device=device,
            overwrite=overwrite,
            verbose=verbose,
        )

        self.api_token = api_token or os.environ.get('HF_TOKEN')

        self.model_name = model_name

        torch_dtype = torch.float16 if device == 'cuda' else torch.float32

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            token=api_token,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        self.model.to(device)

        self.processor = AutoProcessor.from_pretrained(model_name)

        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=30,
            torch_dtype=torch_dtype,
            device=device,
        )

    def inference(
            self,
            audio_path: Union[str, Path],
            transcription_path: Union[str, Path],
            task: str = "transcribe",
            language: Optional[str] = None,
    ):
        """
        Whisper's backed transcription method.

        :param audio_path: Path to the audio track.
        :param transcription_path: Path to the transcription file.
        :param task: Whether to "transcribe" or "translate".
        :param language: Country-code string of the spoken language.
        :return:
        """
        if task not in ["transcribe", "translate"]:
            raise NameError(f"task should be 'transcribe' or 'translate', got '{task}' instead")

        audio_path = Path(audio_path)
        transcription_path = Path(transcription_path)

        if self.verbose:
            print(f"using hugging face")
            print(f"model name      =   {self.model_name}")
            print(f"task            =   {task}")
            if language is not None:
                print(f"language        =   {language}")
            print(f"audio           =   {audio_path}")
            print(f"transcription   =   {transcription_path}")

        wav.WAVReader.check(path=audio_path)
        vtt.VTTWriter.check(path=transcription_path, overwrite=self.overwrite)

        result = self.pipeline(str(audio_path), return_timestamps=True,
                               generate_kwargs={"language": language, "task": task})

        segments = [{'start': chunk['timestamp'][0], 'end': chunk['timestamp'][1], 'text': chunk['text']} for chunk in
                    result['chunks']]
        vtt.VTTWriter.write(
            segments=segments, path=transcription_path, overwrite=self.overwrite
        )
