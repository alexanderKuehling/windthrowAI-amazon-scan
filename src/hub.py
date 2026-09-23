"""Hugging Face Hub access for the two production model checkpoints.

Mirrors how windthrowAI-classification-gradio / windthrowAI-segmentation-gradio
already pull from these same repos, but pins an explicit commit revision instead
of tracking `main` HEAD — resolved once per scan (see scripts/resolve_model_versions.py)
so all ~116 parallel array tasks in slurm/01_inference_array.sh use the exact same
checkpoint even if someone promotes a new one mid-scan.
"""

import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HF_WRITE_TOKEN")


def resolve_revision(repo_id: str) -> str:
    """Return the current HEAD commit sha of a model repo."""
    return HfApi().model_info(repo_id, token=_TOKEN).sha


def download_checkpoint(
    repo_id: str,
    filename: str,
    revision: str,
    local_files_only: bool = False,
) -> Path:
    """Resolve a pinned checkpoint file to a local path via the HF cache.

    `local_files_only=True` (used by every array task) fails loudly instead of
    silently falling back to a live download if the HF cache isn't actually
    shared across cluster nodes.
    """
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_files_only=local_files_only,
            token=_TOKEN,
        )
    )
