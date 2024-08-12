# Usage

## Concept

`psifx` is a Python package that can be used both as a library,
```python
from psifx.audio.diarization.pyannote.tool import PyannoteDiarizationTool

# Parameterize a tool w/ specific settings, such as choosing the underlying neural network, etc.
tool = PyannoteDiarizationTool(...)
# Run the inference method on a given data, here it will be an audio track for example.
tool.inference(...)
```
But it can also come with its own CLI, that can be run directly in a terminal,
```bash
psifx audio diarization pyannote inference --audio /path/to/audio.wav --diarization /path/to/diarization.rttm
```

## Examples

```bash
psifx video manipulation process --in_video Videos/Left.mp4 --out_video Videos/Left.processed.mp4  --start 18 --end 210 --x_min 1347 --y_min 459 --x_max 2553 --y_max 1898 --overwrite
psifx video manipulation process --in_video Videos/Right.mp4 --out_video Videos/Right.processed.mp4  --start 18 --end 210 --x_min 1358 --y_min 435 --x_max 2690 --y_max 2049 --overwrite
```

### [Audio](audio.md)

### [Video](video.md)

### [Text](text.md)