# Text Processing Guide

## Model
Models from Hugging Face, Ollama, OpenAI, and Anthropic are all supported.
Hugging Face and Ollama models are free of charge and can be run locally, while OpenAI and Anthropic models are hosted externally, requiring an API key for access.
Keep in mind that using non-local models may raise privacy concerns.

### Requirements
- **Hugging Face**:
To download models from Hugging Face, you might need a Hugging Face token. Go to [Hugging Face](https://huggingface.co/join) to generate one.
- **Ollama**:
If you're using Ollama with Docker, no additional setup is required.
Otherwise, install Ollama locally by following the instructions on [Ollama](https://github.com/ollama/ollama).
For Linux, it should be as simple as:
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```
- **OpenAI**: 
To access OpenAI models, create an OpenAI account and get an API key.
- **Anthropic**:
To use Anthropic models, sign up at [Anthropic’s Console](https://console.anthropic.com/) and obtain an API key.

### Command Line Arguments


When using a Text tool, you can configure the model with the following command line arguments:

- **`--provider`**: This specifies the model provider you want to use. Options include:
  - `hf` (short for Hugging Face)
  - `ollama`
  - `openai`
  - `anthropic`
  
- **`--model`**: This is the name of the specific language model you'd like to use. Make sure the model is compatible with the provider you’ve selected.

- **`--model_config`**: You may use this optional argument to point to a `.yaml` configuration file. In this file you can set specific runtime settings (e.g., temperature), as well as the model and provider (so you don't have to specify them each time you run a command).
- **`--api_key`**: If the provider you’re using requires a subscription or token for access, supply the API key here. You can also set this as an environment variable instead of including it in the command line.

> **Note**: Do not store the API key within the `.yaml` configuration file to better protect it. 

---

## Usage
### Chat
This feature is intended for benchmarking and testing LLMs (either local or 3rd party) 
for language processing tasks. The user can interact with an LLM in the command terminal, 
whilst specifying a prompt and an output file directory to store the conversation.
```bash
psifx text chat \
    [--prompt chat_history.txt] \
    [--output file.txt] \
    [--provider ollama] \
    [--model llama3.1] \
    [--model_config model_config.yaml] \
    [--api_key api_key]       
```
- `--prompt`: Prompt or path to a .txt file containing the prompt / chat history.
- `--output`: Path to a .txt save file.
### Instruction
This feature is intended to allow general usage of LLMs (either local or 3rd party) for language processing tasks.

```bash
psifx text instruction \
    --instruction instruction.yaml \
    --input input.txt \
    --output output.txt \
    [--provider ollama] \
    [--model llama3.1] \
    [--model_config model_config.yaml] \
    [--api_key api_key] \
  ```
- `--instruction`: Path to a .yaml file containing the prompt and parser, details below.
- `--input`: Path to the input file.
- `--output`: Path to the output file.

  Supported format combinations:
  - `.txt` input → `.txt` output
  - `.vtt` input → `.txt` output
  - `.csv` input → `.csv` output


> **Note**: The .txt and .vtt formats are suited for simpler use cases. 
> The .csv format, however, allows you to process multiple datas and use complex prompts that combine multiples information.

#### Instruction files
Both the prompt and the parser are specified in a .yaml file.
The model will generate an answer to the prompt, this answer will be parsed by the parser, which you will get as output.

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
    to_lower: True 
```
##### Prompt
The prompt enable you to tell the model what you want, and guide its generation.

```yaml
prompt: |
    system: You are an expert doctor.
    user: You take care of a new patient.
    assistant: What are the patient symptoms?
    user: The patient has the following symptoms {text}.
```
Prompts can be customized with the headers **system**, **user**, and **assistant**.

In prompts **{text}** is a placeholder for the content of a .txt or .vtt files. 

When using .csv files, you will instead use placeholder for the content of columns, specified as **{column_name}**. 
Hence, with .csv file you can have placeholder referring to different elements, i.e., **{city}** **{county}**.
```yaml
prompt: |
    user: A patient stayed in hospital {hospital_name}.
    He was asked to fill in this satisfaction questionnaire: {questionary_content}
    Here are the answers he gave: {patient_answers}
    On a scale out of 10 was the patient satisfied?
    What did he think could be improved?
```
##### Parser
If you need to parse the generation from the model, you can specify a parser in the .yaml file.
```yaml
prompt: |
    user: ...
parser:
    start_after: "ANSWER:" 
    to_lower: True 
    expect:
        - "yes"
        - "no" 
```
- `start_after`: Only keep the message after the last instance of the specified text.
- `to_lower`: If True, change the output to lowercase (takes effect subsequently to  start_after).
- `expect`: When the final output is not one of the expected labels prints an error message.
