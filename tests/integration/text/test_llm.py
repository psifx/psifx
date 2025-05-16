"""Integration tests for LLM providers."""

import os
from unittest.mock import patch

import pytest
import yaml
from anthropic import AuthenticationError as AnthropicAuthenticationError
from openai import AuthenticationError as OpenaiAuthenticationError
from psifx import command

@pytest.mark.integration
def test_text_ollama(output_dir):
    """Integration test with Ollama LLM."""
    # Skip if Ollama dependencies are not installed
    langchain = pytest.importorskip("langchain_core", reason="LangChain not installed")
    langchain_ollama = pytest.importorskip("langchain_ollama", reason="LangChain Ollama not installed")

    instruction_path = os.path.join(output_dir, "e2e_instruction.yaml")
    model_config_path = os.path.join(output_dir, "e2e_model_config.yaml")
    text_path = os.path.join(output_dir, "e2e_text.txt")
    output_path = os.path.join(output_dir, "e2e_output.txt")

    instruction = {
        "prompt": "user: Of which color is this flower : {text} ?"
    }
    with open(instruction_path, 'w') as f:
        yaml.dump(instruction, f)

    model_config = {
        "provider": "ollama",
        "model": "qwen3:0.6b"
    }
    with open(model_config_path, 'w') as f:
        yaml.dump(model_config, f)

    with open(text_path, 'w') as f:
        f.write("echinacea")

    with patch("sys.argv",
               ["psifx", "text", "instruction", "--instruction", instruction_path, "--input", text_path, "--output",
                output_path, "--model_config", model_config_path]):
        command.main()

    assert os.path.exists(output_path), "Ollama LLM processing failed"

    # Check if the output file has content
    with open(output_path, 'r') as f:
        content = f.read()

    assert len(content.strip()) > 0, "Ollama output file is empty"


@pytest.mark.integration
def test_text_hugging_face(output_dir):
    """Integration test with Hugging Face LLM."""
    # Skip if HF dependencies are not installed
    langchain = pytest.importorskip("langchain_core", reason="LangChain not installed")
    transformers = pytest.importorskip("transformers", reason="Transformers not installed")

    instruction_path = os.path.join(output_dir, "e2e_instruction.yaml")
    model_config_path = os.path.join(output_dir, "e2e_model_config.yaml")
    text_path = os.path.join(output_dir, "e2e_text.txt")
    output_path = os.path.join(output_dir, "e2e_output.txt")

    instruction = {
        "prompt": "user: Of which color is this flower : {text} ?"
    }
    with open(instruction_path, 'w') as f:
        yaml.dump(instruction, f)

    model_config = {
        "provider": "hf",
        "model": "Qwen/Qwen3-0.6B"
    }
    with open(model_config_path, 'w') as f:
        yaml.dump(model_config, f)

    with open(text_path, 'w') as f:
        f.write("echinacea")

    with patch("sys.argv",
               ["psifx", "text", "instruction", "--instruction", instruction_path, "--input", text_path, "--output",
                output_path, "--model_config", model_config_path]):
        command.main()

    assert os.path.exists(output_path), "Hugging Face LLM processing failed"

    # Check if the output file has content
    with open(output_path, 'r') as f:
        content = f.read()

    assert len(content.strip()) > 0, "Hugging Face output file is empty"


@pytest.mark.integration
def test_text_anthropic(output_dir):
    """Integration test with Anthropic LLM."""
    # Skip if Anthropic dependencies are not installed
    langchain = pytest.importorskip("langchain_core", reason="LangChain not installed")
    langchain_anthropic = pytest.importorskip("langchain_anthropic", reason="LangChain Anthropic not installed")

    instruction_path = os.path.join(output_dir, "e2e_instruction.yaml")
    model_config_path = os.path.join(output_dir, "e2e_model_config.yaml")
    text_path = os.path.join(output_dir, "e2e_text.txt")
    output_path = os.path.join(output_dir, "e2e_output.txt")

    instruction = {
        "prompt": "user: Of which color is this flower : {text} ?"
    }
    with open(instruction_path, 'w') as f:
        yaml.dump(instruction, f)

    model_config = {
        "provider": "anthropic",
        "model": "claude-3-haiku-20240307"
    }
    with open(model_config_path, 'w') as f:
        yaml.dump(model_config, f)

    with open(text_path, 'w') as f:
        f.write("echinacea")

    with pytest.raises(AnthropicAuthenticationError) as exc_info:
        with patch("sys.argv",
                   ["psifx", "text", "instruction", "--instruction", instruction_path, "--input", text_path, "--output",
                    output_path, "--model_config", model_config_path, "--api_key", "sk-ant-fake-key"]):
            command.main()
        assert "invalid x-api-key" in str(exc_info.value)


@pytest.mark.integration
def test_text_openai(output_dir):
    """Integration test with Openai LLM."""
    # Skip if Openai dependencies are not installed
    langchain = pytest.importorskip("langchain_core", reason="LangChain not installed")
    langchain_openai = pytest.importorskip("langchain_openai", reason="LangChain Openai not installed")

    instruction_path = os.path.join(output_dir, "e2e_instruction.yaml")
    model_config_path = os.path.join(output_dir, "e2e_model_config.yaml")
    text_path = os.path.join(output_dir, "e2e_text.txt")
    output_path = os.path.join(output_dir, "e2e_output.txt")

    instruction = {
        "prompt": "user: Of which color is this flower : {text} ?"
    }
    with open(instruction_path, 'w') as f:
        yaml.dump(instruction, f)

    model_config = {
        "provider": "openai",
        "model": "gpt-4o-mini"
    }
    with open(model_config_path, 'w') as f:
        yaml.dump(model_config, f)

    with open(text_path, 'w') as f:
        f.write("echinacea")

    with pytest.raises(OpenaiAuthenticationError) as exc_info:
        with patch("sys.argv",
                   ["psifx", "text", "instruction", "--instruction", instruction_path, "--input", text_path, "--output",
                    output_path, "--model_config", model_config_path, "--api_key", "sk-fake-key"]):
            command.main()
        assert "Incorrect API key provided" in str(exc_info.value)
        assert "sk-fake-key" in str(exc_info.value)