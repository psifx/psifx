# `agents.md` for `psifx`

This file defines how coding agents should operate in this repository for the SAM3 long-video/chunking work and release preparation.

## Core Intent

- Protect established `psifx` protocols and workflows.
- Validate robustness before merging to `psifx/psifx`.
- Avoid regressions in CI, docs, packaging, PyPI, and Docker release paths.

## Non-Negotiables

- Do not deviate from existing repository conventions unless explicitly approved.
- Prefer minimal, targeted changes over broad refactors.
- Never merge or release on assumptions; verify with reproducible checks.
- Treat workflow/release changes as high-risk and gate them with explicit validation.

## Operating Protocol

1. Inspect first, edit second.
2. Before proposing release, verify:
   - tests pass for impacted areas
   - docs reflect actual behavior
   - GitHub workflows still resolve all referenced files and secrets
   - versioning/tagging behavior matches repository automation
3. Surface risks early with concrete evidence (file path, line, command output summary).
4. If a check cannot be run locally (credentials, hardware, network), state that clearly and provide the exact follow-up command.

## Required Validation Gates

### A) Code and Test Gates

- Run targeted SAM3 unit tests first.
- Run broader unit/integration suite appropriate to change scope.
- Run SAM3 chunking/memory validation harness for CPU and GPU paths when available.
- Confirm chunked processing preserves expected output continuity (mask/video frame count and object ID stitching behavior).

### B) Workflow Gates

- Review `.github/workflows/*.yml` touched by release/testing.
- Ensure all workflow-referenced files exist in repo (for example requirement files).
- Confirm trigger semantics are still correct for PR vs `main` vs release events.

### C) Documentation Gates

- Keep `README.md`, `docs/pages/video.md`, and `docker/sam3-validation/README.md` aligned with current CLI and behavior.
- Ensure SAM3 and Samurai docs are not contradictory.
- Keep release/deployment docs aligned with actual workflows.

### D) Release Gates (PyPI + Docker)

- Confirm version source (`psifx.__version__`) is intentionally updated.
- Verify tag/release workflow logic against current remotes and permissions.
- Validate package build locally before release trigger (`python -m build`).
- Validate Docker build path used by CI/release.
- Confirm release order:
  1. merge to upstream `main`
  2. tag/release automation
  3. PyPI publish workflow
  4. Docker publish workflow
  5. docs publish check

## Collaboration and Communication Rules

- Keep plans explicit and phased before execution of high-impact work.
- Report findings by severity first (breaking risk, release risk, documentation risk, then minor).
- Do not hide uncertainty; call it out and propose the smallest safe next step.
- Preserve user-owned changes and avoid destructive git operations unless explicitly requested.

## Definition of Done for Upstream PR Readiness

- SAM3 chunking/memory behavior validated with reproducible evidence.
- Relevant tests pass or remaining gaps are clearly documented.
- Docs and CLI usage are consistent.
- Workflows are syntactically valid and operationally coherent.
- Release checklist for PyPI and Docker is complete and reviewed.
