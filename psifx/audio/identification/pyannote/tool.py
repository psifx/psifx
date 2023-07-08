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
    waveform, sample_rate = Audio(
        sample_rate=sample_rate,
    ).crop(
        file=path,
        segment=Segment(start, end),
    )
    return waveform


class PyannoteIdentificationTool(IdentificationTool):
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

        if api_token is None:
            api_token = "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"

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
        mixed_audio_path = Path(mixed_audio_path)
        diarization_path = Path(diarization_path)
        mono_audio_paths = [Path(path) for path in mono_audio_paths]
        identification_path = Path(identification_path)

        assert mixed_audio_path not in mono_audio_paths
        assert sorted(set(mono_audio_paths)) == sorted(mono_audio_paths)

        wav.WAVReader.check(mixed_audio_path)
        rttm.RTTMReader.check(diarization_path)
        for p in mono_audio_paths:
            wav.WAVReader.check(p)
        json.JSONWriter.check(identification_path)

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
            mapping = {
                speaker_name: mono_audio_name
                for speaker_name, mono_audio_name in zip(
                    speaker_names, mono_audio_names
                )
            }
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


def inference_main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="Path to the audio file.",
    )
    parser.add_argument(
        "--diarization",
        type=Path,
        required=True,
        help="Path to the diarization file.",
    )
    parser.add_argument(
        "--mono_audios",
        nargs="+",
        type=Path,
        required=True,
        help="Paths to the mono audio files.",
    )
    parser.add_argument(
        "--identification",
        type=Path,
        required=True,
        help="Path to the identification file.",
    )
    parser.add_argument(
        "--model_names",
        nargs="+",
        type=str,
        default=[
            "pyannote/embedding",
            "speechbrain/spkrec-ecapa-voxceleb",
        ],
        help="Version number of the pyannote/speaker-diarization model, c.f. https://huggingface.co/pyannote/speaker-diarization/tree/main/reproducible_research",
    )  # TODO
    parser.add_argument(
        "--api_token",
        type=str,
        default=None,
        help="API token for the downloading the models from HuggingFace.",
    )
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

    tool = PyannoteIdentificationTool(
        model_names=args.model_names,
        api_token=args.api_token,
        device=args.device,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )
    tool.inference(
        mixed_audio_path=args.audio,
        diarization_path=args.diarization,
        mono_audio_paths=args.mono_audios,
        identification_path=args.identification,
    )
    del tool
