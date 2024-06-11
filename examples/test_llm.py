from unittest.mock import patch
from psifx import command

"""
COMMAND EXAMPLES
"""

"""
LLM
"""

# Print docs about ollama parameters
args = ["main", "text", "llm", "ollama"]
# Print docs about hf parameters
args = ["main", "text", "llm", "hf"]

"""
CHAT
"""

# Use Ollama llama3 with a prompt given by a string, alternatively with a .txt
args = ["main", "text", "chat", "--model", "llama3", "--prompt", "You are a martian chatbot and you are blue"]
args = ["main", "text", "chat", "--model", "llama3", "--prompt", "chat_history.txt"]

# Chose any specification for the llm with a .json config file
args = ["main", "text", "chat", "--model", "ollama_llama3_cpu.json", "--prompt", "template.txt"]


"""
TASC SEGMENTING
"""

# Segmenting for .vtt files
args = ["main", "text", "tasc", "segment", "--model", "llama3", "--speaker", "student_normalized.wav",
        "--transcription", "transcription.vtt",
        "--segmentation", "segmentation.vtt", "--overwrite"]

# Segmenting for .csv files
args = ["main", "text", "tasc", "segment", "--model", "llama3", "--transcription", "transcription.csv",
        "--segmentation", "segmentation.csv", "--overwrite"]


"""
INSTRUCTION
"""

args = ["main", "text", "instruction", "--model", "llama3", "--input", "input.csv",
        "--output", "output.csv", "--overwrite"]


with patch("sys.argv", args):
    command.main()
