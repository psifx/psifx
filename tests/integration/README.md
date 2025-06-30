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