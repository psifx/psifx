# Development

## Release Flow (PyPI + Docker)

`psifx` release publishing is automated through GitHub Actions:

1. `create-tag-and-release.yml` runs on push to `main`.
2. If `psifx.__version__` differs from the latest tag, it creates a new git tag and GitHub Release.
3. Publishing a release triggers:
   - `build-and-publish-pypi.yml`
   - `build-and-push-docker.yml`
4. Docker publishing updates both `psifx/psifx:<version>` and `psifx/psifx:latest`.
   Documentation is built and deployed only after the Docker build succeeds, using that version's image.

The version source is `psifx/__init__.py` (`psifx.__version__`); package metadata reads it from there.
A code-only merge with no version change does not publish a new release.
Both publishers check that the release tag matches the package version.
Version 0.2.4 introduces SAM3 identity correction and includes `psifx[reid]` in Docker.

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
python -m twine check dist/*
```

Install `build` and `twine` in a separate packaging environment. CI builds and checks the distributions
and runs the Docker test matrix on Python 3.10, 3.11, and 3.12. Changes to requirements, packaging,
Docker, workflows, and tests trigger CI as well as library changes.

For identity-correction changes, install `psifx[reid]` and run
`pytest tests/unit/video/tracking/test_sam3_tool.py tests/integration/video/test_sam3_reid.py`.
The OSNet integration test uses the repository's real two-person clip with a deliberately fragmented
identity. It downloads the small OSNet checkpoint on first use and needs no gated SAM3 weights.
Full SAM3 inference additionally needs an authorized Hugging Face token.
Review overlays and frame continuity on CPU and GPU when available.

## Required GitHub Secrets and Environments

- `ACCESS_TOKEN`: used by the GitHub Release creation step. It needs repository access and permission
  to create releases (Contents read/write for a fine-grained token), and must be a token type allowed
  by the organization. The workflow checks authentication before creating the tag.
- `DOCKERHUB_TOKEN`: used by Docker publish workflow.
- `HF_TOKEN`: used by test workflow and SAM3 validation paths.
- `pypi` environment with trusted publishing enabled for `psifx`.

After merging, check the tag/release, PyPI, Docker, and documentation jobs. A successful source merge
alone does not mean the public artifacts have been updated. Verify the PyPI version, both Docker tags,
and the deployed documentation before announcing the release.

## Manual Docker Build (Local Validation)

```bash
docker build \
  --build-arg PYTHON_VERSION=3.11 \
  -t psifx:local .
```
