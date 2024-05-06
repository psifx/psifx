"""transcription tool."""

from typing import Union

from pathlib import Path

import pandas as pd

from psifx.tool import Tool
from psifx.io import vtt, rttm, json


class TranscriptionTool(Tool):
    """
    Base class for transcription tools.
    """

    def inference(
        self,
        audio_path: Union[str, Path],
        transcription_path: Union[str, Path],
    ):
        """
        Template of the inference method.

        :param audio_path: Path to the audio track.
        :param transcription_path: Path to the transcription file.
        :return:
        """
        audio_path = Path(audio_path)
        transcription_path = Path(transcription_path)

        # audio = load(audio_path)
        # audio = pre_process_func(audio)
        # transcription = model(audio)
        # transcription = post_process_func(transcription)
        # write(transcription, transcription_path)

        raise NotImplementedError

    def enhance(
        self,
        transcription_path: Union[str, Path],
        diarization_path: Union[str, Path],
        identification_path: Union[str, Path],
        enhanced_transcription_path: Union[str, Path],
    ):
        """
        Enhances an audio transcription by fusing the transcribed audio with inferred speaker diarization and identification.

        :param transcription_path: Path to the transcription file.
        :param diarization_path:  Path to the diarization file.
        :param identification_path:  Path to the identification file.
        :param enhanced_transcription_path:  Path to the enhanced diarization file.
        :return:
        """
        transcription_path = Path(transcription_path)
        diarization_path = Path(diarization_path)
        identification_path = Path(identification_path)
        enhanced_transcription_path = Path(enhanced_transcription_path)

        assert transcription_path != enhanced_transcription_path

        if self.verbose:
            print(f"transcription           =   {transcription_path}")
            print(f"diarization             =   {diarization_path}")
            print(f"identification          =   {identification_path}")
            print(f"enhanced_transcription  =   {enhanced_transcription_path}")

        vtt.VTTReader.check(path=transcription_path)
        rttm.RTTMReader.check(path=diarization_path)
        json.JSONReader.check(path=identification_path)
        vtt.VTTWriter.check(path=enhanced_transcription_path, overwrite=self.overwrite)

        transcription = vtt.VTTReader.read(transcription_path)
        transcription = pd.DataFrame.from_records(transcription)

        diarization = rttm.RTTMReader.read(diarization_path)
        diarization = pd.DataFrame.from_records(diarization)
        diarization["end"] = diarization["start"] + diarization["duration"]

        identification = json.JSONReader.read(identification_path)
        mapping = identification["mapping"]

        segments = []
        for transcription_index in range(len(transcription)):
            transcription_row = transcription.iloc[transcription_index]
            highest_iou_index, highest_iou = None, 0.0
            for diarization_index in range(len(diarization)):
                diarization_row = diarization.iloc[diarization_index]
                intersection_start = max(
                    transcription_row["start"], diarization_row["start"]
                )
                intersection_end = min(transcription_row["end"], diarization_row["end"])
                union_start = min(transcription_row["start"], diarization_row["start"])
                union_end = max(transcription_row["end"], diarization_row["end"])
                intersection_duration = max(0.0, intersection_end - intersection_start)
                union_duration = max(0.0, union_end - union_start)
                iou = intersection_duration / union_duration
                if iou > highest_iou:
                    highest_iou_index, highest_iou = diarization_index, iou
            matching_diarization_index = highest_iou_index
            if matching_diarization_index is not None:
                speaker_name = mapping[
                    diarization.iloc[matching_diarization_index]["speaker_name"]
                ]
            else:
                speaker_name = "NA"
            transcription.loc[transcription_index, "speaker"] = speaker_name

            segment = {
                "start": transcription.loc[transcription_index, "start"],
                "end": transcription.loc[transcription_index, "end"],
                "speaker": transcription.loc[transcription_index, "speaker"],
                "text": transcription.loc[transcription_index, "text"],
            }
            segments.append(segment)

        vtt.VTTWriter.write(
            segments=segments,
            path=enhanced_transcription_path,
            overwrite=self.overwrite,
            verbose=self.verbose,
        )
