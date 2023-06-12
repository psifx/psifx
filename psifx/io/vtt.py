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
    x = timestamp.split(":")
    assert len(x) == 3
    y = x[2].split(".")
    assert len(y) == 2
    hours = int(x[0])
    minutes = int(x[1])
    seconds = int(y[0])
    milliseconds = int(y[1])
    return hours * 60.0 * 60.0 + minutes * 60.0 + seconds + milliseconds / 1000.0


class VTTReader(object):
    @staticmethod
    def read(
        path: Union[str, Path],
        verbose: bool = True,
    ) -> Sequence[Dict[str, Union[float, str]]]:
        path = Path(path)

        assert path.suffix == ".vtt"

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


class VTTWriter(object):
    @staticmethod
    def write(
        path: Union[str, Path],
        segments: Sequence[Dict[str, Union[float, str]]],
        overwrite: bool = False,
        verbose: bool = True,
    ):
        path = Path(path)

        assert path.suffix == ".vtt"

        if path.exists():
            if overwrite:
                path.unlink()
            else:
                raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open(mode="w", encoding="utf-8") as file:
            kwargs = dict(sep="\n", end="\n\n", file=file, flush=True)

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
