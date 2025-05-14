# psifx Integration Tests

This directory contains integration tests for the psifx library using pytest.

## Directory Structure

The integration test directory structure mirrors the psifx package structure:

```
integration/     # Integration tests
├── audio/       # Integration tests for psifx.audio
├── video/       # Integration tests for psifx.video
├── text/        # Integration tests for psifx.text
└── data/        # Test data for integration tests
```

## Running Tests

To run all integration tests:

```bash
cd tests/integration/
```

```bash
pytest .
```

To run integration tests for a specific module:

```bash
pytest audio/
```

## Writing Integration Tests

When adding new integration tests:

1. Create test files with the naming pattern `test_*.py`
2. Create test functions with the naming pattern `test_*`
3. Use appropriate fixtures from `conftest.py` when needed
4. Follow the existing test structure and patterns

## Test Data

The `data/` directory contains sample files used for integration testing. These files are:

- `Video.mp4`: A sample video file for testing video and audio processing
- `Audio.wav`: The audio of `Video.mp4`