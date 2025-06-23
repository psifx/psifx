"""Integration tests for LLM providers."""
import os
from pathlib import Path

import pytest
import yaml
from anthropic import AuthenticationError as AnthropicAuthenticationError
from openai import AuthenticationError as OpenaiAuthenticationError
from tests.integration.conftest import run_command


def write_test_files(output_dir: Path, instruction: dict, model_config: dict, input_text: str):
    instruction_path = output_dir / "instruction.yaml"
    model_config_path = output_dir / "model_config.yaml"
    text_path = output_dir / "text.txt"

    instruction_path.write_text(yaml.dump(instruction), encoding="utf-8")
    model_config_path.write_text(yaml.dump(model_config), encoding="utf-8")
    text_path.write_text(input_text, encoding="utf-8")

    return instruction_path, model_config_path, text_path


@pytest.mark.integration
def test_text_ollama(output_dir: Path):
    """Integration test with Ollama LLM."""
    pytest.importorskip("langchain_core", reason="LangChain not installed")
    pytest.importorskip("langchain_ollama", reason="LangChain Ollama not installed")

    instruction = {"prompt": "user: Of which color is this flower : {text} ?"}
    model_config = {"provider": "ollama", "model": "qwen3:0.6b"}
    input_text = "echinacea"

    instruction_path, model_config_path, text_path = write_test_files(output_dir, instruction, model_config, input_text)
    output_path = output_dir / "output.txt"

    run_command(
        "psifx", "text", "instruction",
        "--instruction", instruction_path,
        "--input", text_path,
        "--output", output_path,
        "--model_config", model_config_path
    )

    assert output_path.exists(), "Ollama LLM processing failed"
    assert output_path.read_text().strip(), "Ollama output file is empty"

@pytest.mark.skipif(not os.getenv("HF_TOKEN"), reason="HF_TOKEN not available")
@pytest.mark.integration
def test_text_hugging_face(output_dir: Path):
    """Integration test with Hugging Face LLM."""
    pytest.importorskip("langchain_core", reason="LangChain not installed")
    pytest.importorskip("transformers", reason="Transformers not installed")

    instruction = {"prompt": "user: Of which color is this flower : {text} ?"}
    model_config = {"provider": "hf", "model": "Qwen/Qwen3-0.6B"}
    input_text = "echinacea"

    instruction_path, model_config_path, text_path = write_test_files(output_dir, instruction, model_config, input_text)
    output_path = output_dir / "output.txt"

    run_command(
        "psifx", "text", "instruction",
        "--instruction", instruction_path,
        "--input", text_path,
        "--output", output_path,
        "--model_config", model_config_path
    )

    assert output_path.exists(), "Hugging Face LLM processing failed"
    assert output_path.read_text().strip(), "Hugging Face output file is empty"


@pytest.mark.integration
def test_text_anthropic(output_dir: Path):
    """Integration test with Anthropic LLM."""
    pytest.importorskip("langchain_core", reason="LangChain not installed")
    pytest.importorskip("langchain_anthropic", reason="LangChain Anthropic not installed")

    instruction = {"prompt": "user: Of which color is this flower : {text} ?"}
    model_config = {"provider": "anthropic", "model": "claude-3-haiku-20240307"}
    input_text = "echinacea"

    instruction_path, model_config_path, text_path = write_test_files(output_dir, instruction, model_config, input_text)
    output_path = output_dir / "output.txt"

    with pytest.raises(AnthropicAuthenticationError) as exc_info:
        run_command(
            "psifx", "text", "instruction",
            "--instruction", instruction_path,
            "--input", text_path,
            "--output", output_path,
            "--model_config", model_config_path,
            "--api_key", "sk-ant-fake-key"
        )
    assert "invalid x-api-key" in str(exc_info.value)


@pytest.mark.integration
def test_text_openai(output_dir: Path):
    """Integration test with Openai LLM."""
    pytest.importorskip("langchain_core", reason="LangChain not installed")
    pytest.importorskip("langchain_openai", reason="LangChain Openai not installed")

    instruction = {"prompt": "user: Of which color is this flower : {text} ?"}
    model_config = {"provider": "openai", "model": "gpt-4o-mini"}
    input_text = "echinacea"

    instruction_path, model_config_path, text_path = write_test_files(output_dir, instruction, model_config, input_text)
    output_path = output_dir / "output.txt"

    with pytest.raises(OpenaiAuthenticationError) as exc_info:
        run_command(
            "psifx", "text", "instruction",
            "--instruction", instruction_path,
            "--input", text_path,
            "--output", output_path,
            "--model_config", model_config_path,
            "--api_key", "sk-fake-key"
        )
    assert "Incorrect API key provided" in str(exc_info.value)
    assert "sk-fake-key" in str(exc_info.value)
