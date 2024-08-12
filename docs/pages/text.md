# Text

## Model
Models from Hugging Face, Ollama, OpenAI, and Anthropic are all supported.
Hugging Face and Ollama models are free of charge and can be run locally, while OpenAI and Anthropic models are hosted externally, requiring an API key for access.
Keep in mind that using non-local models may raise privacy concerns.

### Requirements
#### Hugging Face
To download models you will need a Hugging Face token. Go to [Hugging Face](https://huggingface.co/join) to make one.
#### Ollama
You need to install ollama, to do so follow the instructions on [Ollama](https://github.com/ollama/ollama).

For Linux it is a simple as:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
#### Openai
To access OpenAI models you'll need to create an OpenAI account and get an API key.

#### Anthropic
To access Anthropic models you'll need to create an Anthropic account and get an API key
Head to https://console.anthropic.com/ to sign up for Anthropic and generate an API key.

### Configuration file
You can configure the model and its runtime settings through a .yaml file. The configuration file must at least specify the provider (hf, ollama, openai, or anthropic) and the model to be used. Additionally, you can include other parameters such as temperature. Below are example configurations for each provider.

#### Hugging Face
Be sure to include the token field with your HF_TOKEN in the configuration file to automatically download the model.
```yaml
provider: "hf"
model: "mistralai/Mistral-7B-Instruct-v0.2"
token: HF_TOKEN
```
#### Ollama

```yaml
provider: "ollama"
model: "llama3.1"
```

#### Openai
[Optional] You may add the api_key to the yaml file.
```yaml
provider: "openai"
model: "???"
api_key: API_KEY
```
#### Anthropic
[Optional] You may add the api_key to the yaml file.
```yaml
provider: "anthropic"
model: "???"
api_key: API_KEY
```

## Usage
### Chat
This feature is intended to allow general _interactive_ usage of LLMs (either local or 3rd party) for language processing tasks. The user can interact with an LLM in the command terminal, whilst specifying a prompt, previous chat history (optional), and an output file directory to store the chat (optional).
```bash
psifx text chat --llm model_config.yaml \
                [--prompt chat_history.txt] \
                [--output file.txt]
```

### Instruction
This feature is intended to allow general usage of LLMs (either local or 3rd party) for language processing tasks.
```bash
psifx text instruction --llm model_config.yaml \
                       --instruction instruction.yaml \
                       --input input.csv \
                       --output output.csv
```

#### Instruction files
Both the prompt and the parser are specified in a .yaml file.
The process is the following the model will generate an answer to the prompt, this answer will be parsed by the parser, which you will get as output.
```yaml
prompt: |
    user: Here is a semi-structured interview transcript between an
    interviewer denoted ’INTERVIEWER’ and a patient denoted ’PATIENT’
    who is reviewing a mobile app: {text}.
    I am interested in the following question with the desired
    response types in parentheses. Do not make anything up that is
    not in the original transcript. If there is no information to
    answer the question, just write NA. If they barely say anything
    about a question, then the certainty should be very low.
    1) Does the patient find the app useful? (Two integers: Rating out
    of 10 with certainty out of 10 where 10 is maximally certain)
parser:
    kind: "default"
```

Here the {text} key is used to insert an associated csv, vtt or txt file for
integration into the prompt template. 
While vtt and txt file can only replace the field text. For csv any field can be replaced given that the .csv contains the corresponding columns.