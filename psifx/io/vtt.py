"""VTT I/O module."""

from typing import Dict, Sequence, Union

import re
from pathlib import Path
from tqdm import tqdm

TIMEFRAME_LINE_PATTERN = re.compile(
    r"\s*((?:\d+:)?\d{2}:\d{2}.\d{3})\s*"
    + " --> "
    + r"\s*((?:\d+:)?\d{2}:\d{2}.\d{3})\s*"
)
SPEAKER_PATTERN = re.compile(r"<v .+>")


def seconds2timestamp(seconds: float) -> str:
    """
    Converts floating point seconds into timestamp string.

    :param seconds: Floating point seconds.
    :return: Timestamp string.
    """
    assert seconds >= 0
    time = round(seconds * 1000.0)
    hours = time // (60 * 60 * 1000)
    time -= hours * (60 * 60 * 1000)
    minutes = time // (60 * 1000)
    time -= minutes * (60 * 1000)
    seconds = time // 1000
    time -= seconds * 1000
    milliseconds = time
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def timestamp2seconds(timestamp: str) -> float:
    """
    Converts timestamp string into floating point seconds.

    :param timestamp: Timestamp string.
    :return: Floating point seconds.
    """
    parts = timestamp.split(":")
    assert len(parts) == 3
    sub_parts = parts[2].split(".")
    assert len(sub_parts) == 2
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = int(sub_parts[0])
    milliseconds = int(sub_parts[1])
    return hours * 60.0 * 60.0 + minutes * 60.0 + seconds + milliseconds / 1000.0


class VTTReader:
    """
    Safe VTT subtitle reader.
    """

    @staticmethod
    def check(path: Union[str, Path]):
        """
        Checks that a file has of the correct extension and exists.

        :param path: Path to the file.
        :return:
        """
        path = Path(path)
        if path.suffix != ".vtt":
            raise NameError(path)
        if not path.exists():
            raise FileNotFoundError(path)

    @staticmethod
    def read(
        path: Union[str, Path],
        verbose: bool = True,
    ) -> Sequence[Dict[str, Union[float, str]]]:
        """
        Reads and parses a VTT file.

        :param path: Path to the file.
        :param verbose: Verbosity of the method.
        :return: Subtitle segments.
        """
        path = Path(path)
        VTTReader.check(path=path)

        with path.open(encoding="utf-8") as file:
            lines = file.readlines()

        start_indexes = []
        for i, line in enumerate(lines):
            line = line.strip()
            if i == 0:
                assert line == "WEBVTT"
            match = re.match(TIMEFRAME_LINE_PATTERN, line)
            if match is not None:
                start_indexes.append(i)
        end_indexes = start_indexes[1:] + [-1]

        segments = []
        for start_index, end_index in zip(
            tqdm(
                start_indexes,
                desc="Reading",
                disable=not verbose,
            ),
            end_indexes,
        ):
            timestamps = lines[start_index].strip().split(" --> ")
            start, end = [timestamp2seconds(timestamp) for timestamp in timestamps]
            text = ""
            for line in lines[start_index + 1 : end_index]:
                text += line.strip()

            match = re.match(SPEAKER_PATTERN, text)
            if match is not None:
                speaker = match.group(0)
                text = text.replace(speaker, "")
                speaker = speaker.lstrip("<v ").rstrip(">")
            else:
                speaker = None
            segment = {
                "start": start,
                "end": end,
                "speaker": speaker,
                "text": text,
            }
            segments.append(segment)
        return segments


class VTTWriter:
    """
    Safe VTT subtitle writer.
    """

    @staticmethod
    def check(path: Union[str, Path], overwrite: bool = False):
        """
        Checks that a file has of the correct extension and and verifies that we can overwrite it if it exists.

        :param path: Path to the file.
        :return:
        """
        path = Path(path)
        if path.suffix != ".vtt":
            raise NameError(path)
        if path.exists() and not overwrite:
            raise FileExistsError(path)

    @staticmethod
    def write(
        segments: Sequence[Dict[str, Union[float, str]]],
        path: Union[str, Path],
        overwrite: bool = False,
        verbose: bool = True,
    ):
        """
        Writes transcribed audio segments into a VTT subtitle file.

        :param segments: Transcribed audio segments.
        :param path: Path to the file.
        :param overwrite: Whether to overwrite, in case of an existing file.
        :param verbose: Verbosity of the method.
        :return:
        """
        path = Path(path)
        VTTWriter.check(path=path, overwrite=overwrite)

        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and overwrite:
            path.unlink()

        with path.open(mode="w", encoding="utf-8") as file:
            kwargs = {"sep": "\n", "end": "\n\n", "file": file, "flush": True}

            print("WEBVTT", **kwargs)
            for segment in tqdm(
                segments,
                desc="Writing",
                disable=not verbose,
            ):
                start = seconds2timestamp(segment["start"])
                end = seconds2timestamp(segment["end"])
                text = segment["text"].strip().replace("-->", "->")
                speaker = segment.get("speaker", None)

                timeframe = f"{start} --> {end}"
                content = f"<v {speaker}>{text}" if speaker is not None else f"{text}"
                print(timeframe, content, **kwargs)
