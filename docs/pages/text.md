# Text

## Model
Models from Hugging Face, Ollama, OpenAI, and Anthropic are all supported.
Hugging Face and Ollama models are free of charge and can be run locally, while OpenAI and Anthropic models are hosted externally, requiring an API key for access.
Keep in mind that using non-local models may raise privacy concerns.

### Requirements
#### Hugging Face
To download models you will need a Hugging Face token. Go to [Hugging Face](https://huggingface.co/join) to make one.
#### Ollama
If you are using the docker there is nothing to do.
Otherwise you will need to install ollama, please follow the instructions on [Ollama](https://github.com/ollama/ollama).

For Linux it is a simple as:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
#### Openai
To access OpenAI models you'll need to create an OpenAI account and get an API key.

#### Anthropic
To access Anthropic models you'll need to create an Anthropic account and get an API key.
Head to https://console.anthropic.com/ to sign up for Anthropic and generate an API key.

************_**### Model Configuration
When using Text tools you will need to provide a model configuration.
Within the field `--llm` you should indicate the name of a _default_ configuration or the path to a _custom_ configuration file.
The default configurations are **anthropic**, **openai**, **small-local**, **medium-local**, and **large-local**.
Local options are based on Ollama, so make sure you meet the necessary requirements.

You can configure the model and its runtime settings through a .yaml file.
The configuration file must at least specify the provider (hf, ollama, openai, or anthropic) and the model to be used. 
Additionally, you can include other parameters such as temperature. 
Below are example configurations for each provider.

#### Hugging Face
[Optional] You may add your hugging face token to the yaml file, otherwise you will be prompted for it when you download gated models.
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
[Optional] You may add the api_key to the yaml file, otherwise you will be prompted for it.
```yaml
provider: "openai"
model: "???"
api_key: API_KEY
```
#### Anthropic
[Optional] You may add the api_key to the yaml file, otherwise you will be prompted for it.
```yaml
provider: "anthropic"
model: "???"
api_key: API_KEY
```

## Usage
### Chat
This feature is intended to allow general _interactive_ usage of LLMs (either local or 3rd party) for language processing tasks. The user can interact with an LLM in the command terminal, whilst specifying a prompt, previous chat history (optional), and an output file directory to store the chat (optional).
```bash
psifx text chat [--llm model_config.yaml] \
                [--prompt chat_history.txt] \
                [--output file.txt]
```

### Instruction
This feature is intended to allow general usage of LLMs (either local or 3rd party) for language processing tasks.
There is multiple options for --input and --output.
- Both .txt
    ```bash
    psifx text instruction [--llm model_config.yaml] \
                           --instruction instruction.yaml \
                           --input input.txt \
                           --output output.txt
    ```
- From .vtt to .txt
    ```bash
    psifx text instruction [--llm model_config.yaml] \
                           --instruction instruction.yaml \
                           --input input.vtt \
                           --output output.txt
    ```
- Both .csv
    ```bash
    psifx text instruction [--llm model_config.yaml] \
                           --instruction instruction.yaml \
                           --input input.csv \
                           --output output.csv
    ```
The .txt and .vtt options are for simple usages.
With .csv files one can process multiple datas at once and use complex prompts mixing multiples informations together.

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
Prompts can be customized with the headers **system:**, **user:**, and **assistant:**.

In prompts **{text}** is a placeholder for the content of a .txt or .vtt files. 

When using .csv files, you will instead use placeholder for the content of columns, specified as **{column_name}**. 
Hence with .csv file you can have placeholder refering to different elements, i.e., **{city}** **{county}**.
```yaml
prompt: |
    user: A patient stayed in hospital {hospital_name}.
    He was asked to fill in this satisfaction questionnaire: {questionary_content}
    Here are the answers he gave: {patient_answers}
    On a scale out of 10 was the patient satisfied?
    What did he think could be improved?
```
##### Parser
If you need to parse the generation from the model, you can specify a parser in the .yaml file
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
You have the following options:
- start_after: Only keep the message after the last instance of the specified text.
- to_lower: If True, change the output to lowercase (takes effect subsequently to  start_after).
- expect: When the final output is not one of the expected labels prints an error message.
