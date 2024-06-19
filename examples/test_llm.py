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
# Specify llm with .yaml config file
# Directly give the prompt or load it from a .txt file
args = ["main", "text", "chat", "--llm", "llama3:instruct.yaml", "--prompt",
        "You are a martian chatbot and you are blue"]
args = ["main", "text", "chat", "--llm", "llama3:instruct.yaml", "--prompt", "chat_history.txt"]

"""
INSTRUCTION
"""

args = ["main", "text", "instruction", "--llm", "llama3:instruct.yaml", "--input", "input.csv",
        "--output", "output.csv", "--overwrite", "--instruction", "wrestling_instruction.yaml"]

"""
TASC SEGMENTING
"""

# Segmenting for .vtt files
args = ["main", "text", "tasc", "segment", "--llm", "llama3:instruct.yaml", "--speaker", "student_normalized.wav",
        "--transcription", "transcription.vtt",
        "--segmentation", "segmentation.vtt", "--overwrite"]

# Segmenting for .csv files
args = ["main", "text", "tasc", "segment", "--llm", "llama3:instruct.yaml", "--transcription", "parsed_reply.csv",
        "--segmentation", "segmentation.csv", "--overwrite", "--verbose"]

with patch("sys.argv", args):
    command.main()
