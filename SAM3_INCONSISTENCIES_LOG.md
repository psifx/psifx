# SAM3 Integration Inconsistencies Log

Date: 2026-03-30  
Scope: Initial repository + README review before full containerized execution tests.

## Legend
- `Confirmed`: Reproduced directly from current code/docs.
- `Suspected`: Likely issue found in code path; pending runtime confirmation.

## Findings

1. **CLI naming mismatch (`samurai` vs `sam3`)**
- Status: `Confirmed`
- Evidence:
  - Current command registration is `psifx video tracking sam3 ...` in `psifx/video/tracking/command.py`.
  - Example docs/tests still use `samurai`:
    - `example/README.md` uses `psifx video tracking samurai inference`.
    - `tests/integration/video/test_tracking.py` calls `samurai`.
- Impact:
  - Users following examples/tests will run a non-existent command and fail.

2. **README SAM3 setup section has malformed Markdown formatting**
- Status: `Confirmed`
- Evidence:
  - Stray code-fence markers around header/lines in `README.md` (e.g. ``arXiv ...``` and header ending with ``````).
- Impact:
  - Setup instructions render poorly and are harder for non-technical users to follow.

3. **SAM3 model path configuration requires source-code edit**
- Status: `Confirmed`
- Evidence:
  - `psifx/utils/constants.py` hardcodes `SAM3_PATH = "facebook/sam3"`.
  - README instructs users to manually edit package source to point to a local path.
- Impact:
  - Fragile user flow; difficult for non-technical users and awkward for containerized workflows.
- Note:
  - Environment-variable override is not currently supported.

4. **Crash when object is missing in a frame during mask writing**
- Status: `Confirmed`
- Evidence:
  - In `psifx/video/tracking/sam3/tool.py`, `_build_unified_output` can store `None` for absent object masks.
  - In `_write_masks`, `data_tens.detach().cpu().numpy()` is called before checking for `None`.
  - Isolated runtime reproduction:
    - Calling `_write_masks` with `unified = {0: {0: None}}` raises:
      - `AttributeError: 'NoneType' object has no attribute 'detach'`
- Impact:
  - Tracking may fail at runtime on normal videos where an object disappears briefly.

5. **CPU path likely unstable due forced `bfloat16`**
- Status: `Suspected`
- Evidence:
  - `Sam3TrackingTool` always moves model/session to `dtype=torch.bfloat16` for both CPU and CUDA.
- Impact:
  - CPU inference may error or behave inconsistently depending on hardware/operator support.

6. **Conda setup can fail due missing Terms-of-Service acceptance step**
- Status: `Confirmed`
- Evidence:
  - Fresh Miniconda build failed during `conda create` with:
    - `CondaToSNonInteractiveError: Terms of Service have not been accepted`
  - Current README Conda instructions do not include channel ToS acceptance.
- Impact:
  - Non-interactive installs (especially Docker/CI) can fail before environment creation.

7. **Default SAM3 inference path fails without explicit HF authentication**
- Status: `Confirmed`
- Evidence:
  - Running `psifx video tracking sam3 inference ...` with default `SAM3_PATH="facebook/sam3"` returned:
    - `401 Unauthorized`
    - `GatedRepoError` / `Cannot access gated repo ... facebook/sam3`
- Impact:
  - Out-of-the-box user flow fails unless users already have gated-model access and token configuration.
- Note:
  - Error trace is long and technical; no concise user-facing guidance is shown by the command.

8. **SAM3 command does not expose token/path flags (unlike other HF-backed modules)**
- Status: `Confirmed`
- Evidence:
  - `psifx video tracking sam3 inference` arguments do not include `--api_token` or `--model_path`.
  - Comparison: pyannote command includes `--api_token`.
- Impact:
  - Users must rely on hidden/global setup (editing constants or preconfigured environment) instead of explicit CLI arguments.

9. **Every `psifx` command prints unrelated pyannote/speechbrain warnings**
- Status: `Confirmed`
- Evidence:
  - Even `psifx --help` and SAM3 commands print torchaudio deprecation warnings from pyannote/speechbrain.
- Impact:
  - Adds noisy output and cognitive load for non-technical users.
- Likely cause:
  - Eager top-level imports of audio modules when launching global CLI.

10. **Visualization command can emit a confusing error when mask directory is invalid/missing**
- Status: `Confirmed`
- Evidence:
  - Received `ValueError: Expected .mp4 file, got /tmp/.../masks-cpu` when visualization got a non-existent mask directory path.
- Impact:
  - Error message points to file extension rather than directory existence/emptiness, which can mislead users.

11. **`samurai` references remain in docs/tests despite migration to `sam3`**
- Status: `Confirmed`
- Evidence (`rg -n "samurai"`):
  - `tests/integration/video/test_tracking.py`
  - `docs/pages/video.md`
  - `example/README.md`
- Impact:
  - Users and CI/tests still point to an obsolete command path.
- Dev action:
  - Replace `samurai` usage with `sam3` and refresh surrounding wording.

12. **Tracking documentation no longer matches implemented SAM3 CLI**
- Status: `Confirmed`
- Evidence:
  - `docs/pages/video.md` still documents old Samurai/YOLO options:
    - `--model_size`, `--yolo_model`, `--object_class`, `--max_objects`, `--step`
  - Current `sam3` implementation exposes different flags:
    - `--text_prompt`, `--chunk_size`, `--iou_threshold`, `--device`
- Impact:
  - Users following docs will run invalid/unknown options.

13. **README setup points users to a personal fork instead of canonical repo**
- Status: `Confirmed`
- Evidence:
  - SAM3 setup section uses `git clone https://github.com/BogdanvL/psifx.git`.
- Impact:
  - Non-technical users may end up on a fork with diverging history/instructions.

14. **No practical "mid-size" SAM3 model option exposed for test workflows**
- Status: `Confirmed`
- Evidence:
  - Current SAM3 integration only loads `SAM3_PATH` (default `facebook/sam3`) via hardcoded constant in `psifx/utils/constants.py`.
  - CLI does not expose model-selection flags (`--model_path`, `--model_name`) for SAM3.
  - For this code path, the only lightweight debug alternative we validated is `tiny-random/sam3`; there is no documented built-in "small-but-real" checkpoint option in current repo flow.
- Impact:
  - Teams must choose between very slow/heavy full model runs or low-fidelity random-model smoke tests, making QA validation harder.

15. **End-to-end SAM3 validation is difficult to confirm with current UX/telemetry**
- Status: `Suspected`
- Evidence:
  - Inference processes can run for extended periods with little user-facing progress indication.
  - During long runs, output mask directory may remain empty for a long time, making it hard to distinguish "still processing" from "stalled/failing" without container/process introspection.
- Impact:
  - Non-technical users cannot easily tell whether SAM3 is working, hung, or just slow, which creates repeated restart/debug loops.

16. **Short-clip completion is currently non-deterministic (full model)**
- Status: `Confirmed`
- Evidence:
  - Controlled scripted validation on a 2-frame clip (`example/data/Video.mp4` subset) with full model (`facebook/sam3`) and explicit 300s timeout per device produced:
    - `cuda_exit_code=124`, `cuda_mask_count=0`
    - `cpu_exit_code=124`, `cpu_mask_count=0`
  - In separate ad-hoc runs, GPU also showed a successful completion case for 2 frames (produced `0.mp4` with 2 frames), indicating inconsistent runtime behavior across runs.
- Impact:
  - Users cannot rely on a predictable "quick sanity check" flow for CPU/GPU completion, even on ultra-short clips, without additional stabilization and observability.
