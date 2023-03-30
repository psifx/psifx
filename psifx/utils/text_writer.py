from typing import TextIO

from pathlib import Path
import json

from psifx.utils.timestamp import format_timestamp


class BaseTextWriter:
    suffix: str

    def __call__(self, result: dict, path: Path):
        assert path.suffix == self.suffix

        with path.open("w", encoding="utf-8") as file:
            self.write_result(result, file=file)

    def write_result(self, result: dict, file: TextIO):
        raise NotImplementedError


class RTTMWriter(BaseTextWriter):
    suffix: str = ".rttm"

    def write_result(self, result: dict, file: TextIO):
        for segment in result["segments"]:
            start = segment["start"]
            end = segment["end"]
            if end > start:
                duration = end - start
            else:
                duration = 0
            print(
                f"SPEAKER {segment['uri']} 1 {start:.3f} {duration:.3f} <NA> <NA> {segment['label']} <NA> <NA>",
                file=file,
                flush=True,
            )


class TXTWriter(BaseTextWriter):
    suffix: str = ".txt"

    def write_result(self, result: dict, file: TextIO):
        for segment in result["segments"]:
            print(
                segment["text"].strip(),
                file=file,
                flush=True,
            )


class VTTWriter(BaseTextWriter):
    suffix: str = ".vtt"

    def write_result(self, result: dict, file: TextIO):
        print("WEBVTT\n", file=file)
        for segment in result["segments"]:
            print(
                f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
                f"{segment['text'].strip().replace('-->', '->')}\n",
                file=file,
                flush=True,
            )


class SRTWriter(BaseTextWriter):
    suffix: str = ".srt"

    def write_result(self, result: dict, file: TextIO):
        for i, segment in enumerate(result["segments"], start=1):
            # write srt lines
            print(
                f"{i}\n"
                f"{format_timestamp(segment['start'], always_include_hours=True, decimal_marker=',')} --> "
                f"{format_timestamp(segment['end'], always_include_hours=True, decimal_marker=',')}\n"
                f"{segment['text'].strip().replace('-->', '->')}\n",
                file=file,
                flush=True,
            )


class TSVWriter(BaseTextWriter):
    """
    Write a transcript to a file in TSV (tab-separated values) format containing lines like:
    <start time in integer milliseconds>\t<end time in integer milliseconds>\t<transcript text>

    Using integer milliseconds as start and end times means there's no chance of interference from
    an environment setting a language encoding that causes the decimal in a floating point number
    to appear as a comma; also is faster and more efficient to parse & store, e.g., in C++.
    """

    suffix: str = ".tsv"

    def write_result(self, result: dict, file: TextIO):
        print("start", "end", "text", sep="\t", file=file)
        for segment in result["segments"]:
            print(round(1000 * segment["start"]), file=file, end="\t")
            print(round(1000 * segment["end"]), file=file, end="\t")
            print(
                segment["text"].strip().replace("\t", " "),
                file=file,
                flush=True,
            )


class JSONWriter(BaseTextWriter):
    suffix: str = ".json"

    def write_result(self, result: dict, file: TextIO):
        json.dump(result, file)
