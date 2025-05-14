# Psifx Tests

This directory contains tests for the psifx library using pytest.

## Directory Structure

The test directory structure is organized as follows:

```
tests/
├── unit/             # Unit tests
├── integration/      # Integration tests @integration
└── test_structure.py # Tests for the structure of psifx @structure
```

- **Unit Tests**: Test individual functions and classes in isolation, often using mocks for dependencies.
- **Integration Tests**: Test how different components of the library work together, using real data and minimal
  mocking.

Integration tests are marked with the `@pytest.mark.integration` decorator.

## Running Tests

To run only integration tests:

```bash
pytest -m "integration"
```

To run tests with coverage report:

```bash
pytest --cov=psifx
```

To run tests for a specific module:

```bash
pytest tests/integration/io/
```

## Writing Tests

When adding new tests:

1. Create test files with the naming pattern `test_*.py`
2. Create test functions with the naming pattern `test_*`
3. Use appropriate fixtures from `conftest.py` when needed
4. Follow the existing test structure and patterns
