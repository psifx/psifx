# Development

## Release Flow (PyPI + Docker)

`psifx` release publishing is automated through GitHub Actions:

1. `create-tag-and-release.yml` runs on push to `main`.
2. If `psifx.__version__` differs from the latest tag, it creates a new git tag and GitHub Release.
3. Publishing a release triggers:
   - `build-and-publish-pypi.yml`
   - `build-and-push-docker.yml`

## Pre-Release Checklist

Before merging a release PR into `main`:

```bash
# Ensure version is intentionally set
python - <<'PY'
import psifx
print(psifx.__version__)
PY

# Run tests
pytest -rs tests

# Build package artifacts
python -m build
```

## Required GitHub Secrets and Environments

- `ACCESS_TOKEN`: used by tag/release workflow.
- `DOCKERHUB_TOKEN`: used by Docker publish workflow.
- `HF_TOKEN`: used by test workflow and SAM3 validation paths.
- `pypi` environment with trusted publishing enabled for `psifx`.

## Manual Docker Build (Local Validation)

```bash
docker build \
  --build-arg PYTHON_VERSION=3.11 \
  -t psifx:local .
```
