"""pyannote speaker identification tool."""

from typing import Union, Optional, Sequence

from itertools import permutations
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
from torch import Tensor

from pyannote.audio import Audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.core import Segment

from psifx.audio.identification.tool import IdentificationTool
from psifx.io import rttm, json, wav


def cropped_waveform(
    path: Union[str, Path],
    start: float,
    end: float,
    sample_rate: int = 16000,
) -> Tensor:
    """
    Crops an audio track and returns its corresponding waveform.

    :param path: Path to the audio track.
    :param start: Start of segment in seconds.
    :param end: End of the segment in seconds.
    :param sample_rate: Sample rate of the audio track.
    :return: Tensor containing the waveform of the audio segment.
    """
    waveform, sample_rate = Audio(
        sample_rate=sample_rate,
    ).crop(
        file=path,
        segment=Segment(start, end),
        mode="pad",
    )
    return waveform


class PyannoteIdentificationTool(IdentificationTool):
    """
    pyannote speaker identification tool.

    :param model_names: The names of the models to use.
    :param api_token: The HuggingFace API token to use.
    :param device: The device where the computation should be executed.
    :param overwrite: Whether to overwrite existing files, otherwise raise an error.
    :param verbose: Whether to execute the computation verbosely.
    """

    def __init__(
        self,
        model_names: Sequence[str],
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

        self.api_token = api_token
        self.models = {
            name: PretrainedSpeakerEmbedding(
                embedding=name,
                device=torch.device(device),
                use_auth_token=api_token,
            )
            for name in model_names
        }

    def inference(
        self,
        mixed_audio_path: Union[str, Path],
        diarization_path: Union[str, Path],
        mono_audio_paths: Sequence[Union[str, Path]],
        identification_path: Union[str, Path],
    ):
        """
        pyannote's backed inference method.

        :param mixed_audio_path: Path to the mixed audio track.
        :param diarization_path: Path to the diarization file.
        :param mono_audio_paths: Path to the mono audio tracks.
        :param identification_path: Path to the identification file.
        :return:
        """
        mixed_audio_path = Path(mixed_audio_path)
        diarization_path = Path(diarization_path)
        mono_audio_paths = [Path(path) for path in mono_audio_paths]
        identification_path = Path(identification_path)

        assert mixed_audio_path not in mono_audio_paths
        assert sorted(set(mono_audio_paths)) == sorted(mono_audio_paths)

        wav.WAVReader.check(path=mixed_audio_path)
        rttm.RTTMReader.check(path=diarization_path)
        for path in mono_audio_paths:
            wav.WAVReader.check(path=path)
        json.JSONWriter.check(path=identification_path, overwrite=self.overwrite)

        segments = rttm.RTTMReader.read(path=diarization_path, verbose=self.verbose)
        dataframe = pd.DataFrame.from_records(segments)
        dataframe["end"] = dataframe["start"] + dataframe["duration"]

        for name, model in tqdm(
            self.models.items(),
            desc="Processing",
            disable=not self.verbose,
        ):
            distances = []
            valids = []
            for index in tqdm(
                range(dataframe.shape[0]),
                desc="Model Embedding",
                disable=not self.verbose,
                leave=False,
            ):
                row = dataframe.iloc[index]
                if row["duration"] < 0.300:
                    distance = np.nan
                else:
                    mixed_embedding = model(
                        waveforms=cropped_waveform(
                            path=mixed_audio_path,
                            start=row["start"],
                            end=row["end"],
                        )[None, ...]
                    )
                    mono_embeddings = np.concatenate(
                        [
                            model(
                                waveforms=cropped_waveform(
                                    path=path,
                                    start=row["start"],
                                    end=row["end"],
                                )[None, ...]
                            )
                            for path in mono_audio_paths
                        ]
                    )
                    delta = mixed_embedding - mono_embeddings
                    distance = np.linalg.norm(delta, ord=2, axis=-1)

                valid = np.isfinite(distance).all()

                distances.append(distance)
                valids.append(valid)

            dataframe[f"distance_{name}"] = distances
            dataframe[f"valid_{name}"] = valids

        for name in self.models.keys():
            dataframe.drop(
                index=dataframe[~dataframe[f"valid_{name}"]].index,
                inplace=True,
            )

        mono_audio_names = [path.name for path in mono_audio_paths]

        best_mapping = None
        best_agreement = 0.0
        for speaker_names in tqdm(
            permutations(pd.Categorical(dataframe["speaker_name"]).categories.tolist()),
            desc="Voting",
            disable=not self.verbose,
        ):
            mapping = dict(zip(speaker_names, mono_audio_names))
            speaker_ids = np.stack(
                [
                    mono_audio_names.index(mapping[name])
                    for name in dataframe["speaker_name"].values
                ]
            )
            model_agreements = []
            for name in self.models.keys():
                distances = np.stack(dataframe[f"distance_{name}"].values)
                closest_ids = distances.argmin(axis=-1)
                model_agreement = (speaker_ids == closest_ids).mean()
                model_agreements.append(model_agreement)
            average_agreement = np.stack(model_agreements).mean()
            if average_agreement > best_agreement:
                best_agreement = average_agreement
                best_mapping = mapping

        data = {
            "mapping": best_mapping,
            "agreement": best_agreement,
        }
        json.JSONWriter.write(
            data=data,
            path=identification_path,
            overwrite=self.overwrite,
            verbose=self.verbose,
        )
