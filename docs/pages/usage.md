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

### Audio

#### Pre-processing

```bash
psifx audio manipulation extraction --video Videos/Left.mp4 --audio Audios/Left.wav
psifx audio manipulation extraction --video Videos/Right.mp4 --audio Audios/Right.wav

psifx audio manipulation mixdown --mono_audios Audios/Right.wav Audios/Left.wav --mixed_audio Audios/Mixed.wav

psifx audio manipulation normalization --audio Audios/Right.wav --normalized_audio Audios/Right.normalized.wav
psifx audio manipulation normalization --audio Audios/Left.wav --normalized_audio Audios/Left.normalized.wav
psifx audio manipulation normalization --audio Audios/Mixed.wav --normalized_audio Audios/Mixed.normalized.wav
```

### Inference

```bash
psifx audio diarization pyannote inference --audio Audios/Mixed.normalized.wav --diarization Diarizations/Mixed.rttm --num_speakers 2 --device cuda

psifx audio identification pyannote inference --mixed_audio Audios/Mixed.normalized.wav --diarization Diarizations/Mixed.rttm --mono_audios Audios/Left.normalized.wav Audios/Right.normalized.wav --identification Identifications/Mixed.json --device cuda

psifx audio transcription whisper inference --audio Audios/Mixed.normalized.wav --transcription Transcriptions/Mixed.vtt --model_name large --language fr --device cuda

psifx audio transcription whisper enhance --transcription Transcriptions/Mixed.vtt --diarization Diarizations/Mixed.rttm --identification Identifications/Mixed.json --enhanced_transcription Transcriptions/Mixed.enhanced.vtt
```

### Visualization

```bash
psifx audio diarization visualization --diarization Diarizations/Mixed.rttm --visualization Visualizations/Mixed.png
```

### Video

#### Inference

```bash
psifx video pose mediapipe inference --video Videos/Right.mp4 --poses Poses/Right.tar.xz --masks Masks/Right.mp4
psifx video pose mediapipe inference --video Videos/Left.mp4 --poses Poses/Left.tar.xz --masks Masks/Left.mp4

psifx video face openface inference --video Videos/Right.mp4 --features Faces/Right.tar.xz
psifx video face openface inference --video Videos/Left.mp4 --features Faces/Left.tar.xz
```

#### Visualization

```bash
psifx video pose mediapipe visualization --video Videos/Right.mp4 --poses Poses/Right.tar.xz --visualization Visualizations/Right.mediapipe.mp4
psifx video pose mediapipe visualization --video Videos/Left.mp4 --poses Poses/Left.tar.xz --visualization Visualizations/Left.mediapipe.mp4

psifx video face openface visualization --video Videos/Right.mp4 --features Faces/Right.tar.xz --visualization Visualizations/Right.openface.mp4
psifx video face openface visualization --video Videos/Left.mp4 --features Faces/Left.tar.xz --visualization Visualizations/Left.openface.mp4
```

### Text

#### LLM Configs
You can set up the llm to use as well as its running configuration in a .yaml file.
All models from hugging face and ollama are supported.
##### Hugging Face
Specify hf as provider.
Make sure to include your HF_TOKEN to automatically download the model.
```yaml
provider: "hf"
model: "mistralai/Mistral-7B-Instruct-v0.2"
quantization: "4bit"
max_new_tokens: 1000
token: HF_TOKEN
```
##### Ollama
You need to install ollama and download the model before using it.
To do so follow the instructions on https://github.com/ollama/ollama.

For Linux it is a simple as:
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull MODEL_NAME
```
Specify ollama as provider.
```yaml
provider: "ollama"
model: "llama3:instruct"
```

#### Chatbot

```bash
psifx text chat --llm model_config.yaml --prompt chat_history.txt
```

#### Instruction

```bash
psifx text instruction --llm model_config.yaml --instruction instruction.yaml --input input.csv --output output.csv
```

##### Instruction files
Both the prompt and the parser are specified in a .yaml file.
```yaml
wrestling:
    parser:
        kind: "default"
        start_flag: "Winner:"
    prompt: |
        user: Here is a challenger {challenger} and an opponent {opponent} which one would win in a professional wrestling match?
        Argue about it.
        Once you have deliberated, write as your last sentence the result of your deliberation in the following format Winner: 'name of the winner'.
        Nothing should follow this last answer.
```