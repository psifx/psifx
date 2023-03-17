from pathlib import Path
import math

import matplotlib.pyplot as plt

import torch
from torch import Tensor

import torchaudio
from torchaudio import transforms as T, functional as F

from pydub import AudioSegment


def plot_waveform(waveform: Tensor, sample_rate: int, title: str = "Waveform"):
    waveform = waveform.detach().cpu().numpy()

    n_channels, n_frames = waveform.shape
    time_axis = torch.linspace(0, (n_frames - 1) / sample_rate, n_frames).cpu().numpy()

    figure, axes = plt.subplots(n_channels, 1)
    if n_channels == 1:
        axes.plot(time_axis, waveform[0], linewidth=1)
        axes.grid(True)
    else:
        for i in range(n_channels):
            axes[i].plot(time_axis, waveform[i], linewidth=1)
            axes[i].grid(True)
    figure.suptitle(title)
    plt.show(block=False)


def plot_spectrogram(spectrogram: Tensor, title: str = "Spectrogram"):
    spectrogram = spectrogram.detach().cpu().numpy()

    n_channels, n_frequencies, n_frames = spectrogram.shape
    figure, axes = plt.subplots(n_channels, 1)
    if n_channels == 1:
        axes.imshow(spectrogram[0], origin="lower", aspect="auto")
        axes.set_xlabel("Windows")
        axes.set_ylabel("Frequency Bins")
    else:
        for i in range(n_channels):
            axes[i].imshow(spectrogram[i], origin="lower", aspect="auto")
            axes[i].set_xlabel("Windows")
            axes[i].set_ylabel("Frequency Bins")
    figure.suptitle(title)
    plt.show(block=False)


def main():
    audio_path = "/home/guillaume/Datasets/UNIL/CH.101/Audios/CH.101.h.wav"
    diarization_path = "/home/guillaume/Datasets/UNIL/CH.101/Diarizations/CH.101.h.rttm"

    audio_path = Path(audio_path)
    diarization_path = Path(diarization_path)

    waveform, sample_rate = torchaudio.load(audio_path)
    n_channels, n_frames = waveform.shape
    print(waveform.shape, waveform.dtype)

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
            speaker_id = int(str(line[7]).replace("SPEAKER_", ""))
            segment = {
                "time_start": time_start,
                "time_end": time_end,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "speaker_id": speaker_id,
            }
            segments.append(segment)
    print(len(segments))
    print(segments)

    spectrogram_transform = T.Spectrogram(
        n_fft=2048,
        power=2.0,
        onesided=False,
    )
    db_transform = T.AmplitudeToDB(stype="power", top_db=80.0)

    for i, segment in enumerate(segments):
        print(i, segment)
        local_waveform = waveform[:, segment["frame_start"] : segment["frame_end"]]
        local_spectrogram = spectrogram_transform(local_waveform)
        # print(local_waveform.shape)
        # print(local_spectrogram.shape)
        # print(local_spectrogram.min(), local_spectrogram.max())
        x = local_spectrogram.argmax(dim=-2).median(dim=-1).values
        print(x)

        # plot_waveform(local_waveform, sample_rate)
        # plot_spectrogram(db_transform(local_spectrogram))
        # input("Waiting...")


def main2():
    audio_path = "/home/guillaume/Datasets/UNIL/CH.101/Audios/CH.101.h.wav"
    diarization_path = "/home/guillaume/Datasets/UNIL/CH.101/Diarizations/CH.101.h.rttm"

    audio_path = Path(audio_path)
    diarization_path = Path(diarization_path)

    waveform, sample_rate = torchaudio.load(audio_path)
    n_channels, n_frames = waveform.shape
    print(waveform.shape, waveform.dtype)
    print(waveform.min(), waveform.max())

    # Read and parse the segments from file.
    segments = []
    with diarization_path.open() as file:
        for line in file.readlines():
            print(line)
            line = line.split(" ")
            time_start = float(line[3])
            frame_start = int(sample_rate * time_start)
            time_duration = float(line[4])
            frame_duration = int(sample_rate * time_duration)
            frame_end = frame_start + frame_duration
            time_end = float(frame_end / sample_rate)
            speaker_id = int(str(line[7]).replace("SPEAKER_", ""))
            segment = {
                "time_start": time_start,
                "time_end": time_end,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "speaker_id": speaker_id,
            }
            segments.append(segment)
    print(segments)

    # Create a mask whose indexes are the speaker indexes.
    speaker_mask = torch.full_like(waveform, fill_value=-1, dtype=torch.int64)
    for segment in segments:
        speaker_mask[:, segment["frame_start"] : segment["frame_end"]] = segment[
            "speaker_id"
        ]
    print(speaker_mask.shape, speaker_mask.dtype)

    speaker_ids = speaker_mask.unique()[1:]

    # Compute a heuristic to determine who the main speaker is.
    speaker_magnitudes = torch.empty_like(speaker_ids, dtype=waveform.dtype)
    for i, speaker_id in enumerate(speaker_ids):
        magnitude = waveform.abs().square()
        mask = speaker_mask == speaker_id
        magnitude[~mask] = math.nan
        median_magnitude = magnitude.nanmedian(dim=-1, keepdim=True).values.mean()
        speaker_magnitudes[i] = median_magnitude

    print(speaker_ids)
    print(speaker_magnitudes)

    main_speaker = speaker_ids[speaker_magnitudes.argmax()]
    print(main_speaker)


if __name__ == "__main__":
    main()
