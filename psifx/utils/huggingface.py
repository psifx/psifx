"""Compatibility helpers for Hugging Face integrations."""

from __future__ import annotations

import importlib
import inspect
from functools import wraps


def patch_hf_hub_download_use_auth_token() -> None:
    """
    Patch ``hf_hub_download`` so callers using ``use_auth_token=...`` still work.

    Some dependencies still call ``hf_hub_download(..., use_auth_token=...)`` while
    newer ``huggingface_hub`` versions expect ``token=...``.
    """
    try:
        import huggingface_hub
    except Exception:
        return

    hf_hub_download = getattr(huggingface_hub, "hf_hub_download", None)
    if hf_hub_download is None:
        return

    try:
        signature = inspect.signature(hf_hub_download)
    except (TypeError, ValueError):
        signature = None

    if signature is not None and "use_auth_token" in signature.parameters:
        return

    @wraps(hf_hub_download)
    def compat_hf_hub_download(*args, use_auth_token=None, **kwargs):
        if use_auth_token is not None and "token" not in kwargs:
            kwargs["token"] = use_auth_token
        return hf_hub_download(*args, **kwargs)

    huggingface_hub.hf_hub_download = compat_hf_hub_download

    # pyannote imports hf_hub_download directly in these modules.
    for module_name in (
        "pyannote.audio.core.pipeline",
        "pyannote.audio.core.model",
    ):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        if hasattr(module, "hf_hub_download"):
            setattr(module, "hf_hub_download", compat_hf_hub_download)
