"""
Resolve and pin the production model checkpoints for one scan.

Run once per scan, before the inference array job (slurm/00_generate_grid.sh
calls this right after generate_grid.py). Resolves the current HEAD commit sha
of both HF Hub model repos, downloads both files once to warm the shared HF
cache, and writes model_versions.json at the repo root.

Every array task in scripts/run_inference.py then reads that file and
downloads with local_files_only=True — so the whole scan uses one pinned
checkpoint pair, even if someone promotes a new model to HF mid-scan.

    python scripts/resolve_model_versions.py

No error handling around the HF calls on purpose: a network/auth failure here
must stop the pipeline (slurm/00_generate_grid.sh has `set -euo pipefail`) —
never fall back silently to an unpinned or stale model.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    CLASSIFIER_FILENAME,
    CLASSIFIER_REPO,
    MODEL_VERSIONS_FILE,
    SEGMENTER_FILENAME,
    SEGMENTER_REPO,
)
from src.hub import download_checkpoint, resolve_revision


def _resolve_and_warm(repo_id: str, filename: str) -> dict:
    revision = resolve_revision(repo_id)
    download_checkpoint(repo_id, filename, revision, local_files_only=False)
    return {"repo_id": repo_id, "filename": filename, "revision": revision}


def main():
    print("Resolving production model versions from Hugging Face Hub …")
    classifier = _resolve_and_warm(CLASSIFIER_REPO, CLASSIFIER_FILENAME)
    print(f"  classifier: {classifier['repo_id']} @ {classifier['revision']}")
    segmenter = _resolve_and_warm(SEGMENTER_REPO, SEGMENTER_FILENAME)
    print(f"  segmenter:  {segmenter['repo_id']} @ {segmenter['revision']}")

    MODEL_VERSIONS_FILE.write_text(json.dumps({
        "resolved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "classifier":  classifier,
        "segmenter":   segmenter,
    }, indent=2))
    print(f"wrote {MODEL_VERSIONS_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
