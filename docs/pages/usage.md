# Usage

## Concept

`psifx` is a versatile Python package that can be used both as a library within Python code or as a command-line tool for direct execution.

### Library Usage
As a library, `psifx` provides tools for various tasks, including audio processing, video manipulation, and text processing. For example, you can use it for speaker diarization in Python by importing the necessary modules and specifying parameters programmatically:

```python
from psifx.audio.diarization.pyannote.tool import PyannoteDiarizationTool

# Configure a tool with specific settings, such as selecting a neural network model.
tool = PyannoteDiarizationTool(model_name="pyannote/speaker-diarization-3.1")

# Run the inference method on an audio track for speaker segmentation.
tool.inference(audio_path="/path/to/audio.wav", 
               diarization_path="/path/to/diarization.rttm", 
               num_speakers=2)
```

### Command-Line Interface (CLI)
`psifx` also includes a powerful CLI for running commands directly in the terminal. This can simplify workflows by allowing users to specify tasks and parameters without writing any Python code.

For example, to run speaker diarization on an audio file:

```bash
psifx audio diarization pyannote inference \
    --audio /path/to/audio.wav \
    --diarization /path/to/diarization.rttm \
    --num_speakers 2
```

## Additional Resources

For detailed command usage in specific areas, see:

- [Audio Processing Guide](audio.md)
- [Video Processing Guide](video.md)
- [Text Processing Guide](text.md)

These guides provide in-depth instructions for working with different data types in `psifx`, from pre-processing to inference and visualization.