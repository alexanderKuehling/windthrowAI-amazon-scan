"""
One-off migration: re-encode the inputs/ already published in the HF dataset from
raw float32/LZW (~2.0 MB/tile) to the compact uint16 web encoding (~0.6 MB/tile),
so the Expert-Review tab stops waiting seconds per patch.

    python scripts/recompress_hf_inputs.py            # dry run: report the size win
    python scripts/recompress_hf_inputs.py --upload   # re-encode and push inputs/ back

Re-running is safe: already-encoded tiles (uint16) are skipped. Newly exported
runs need no migration — export_for_frontend.py writes this encoding directly.

Requires: rasterio, huggingface_hub, and `huggingface-cli login` (or HF_TOKEN)
with write access when --upload is passed.
"""

import argparse
import sys
from pathlib import Path

import rasterio

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.export_for_frontend import HF_REPO_ID, encode_input_for_web  # noqa: E402

WORK_DIR = ROOT / "outputs_final" / "hf_inputs_recompress"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-repo", default=HF_REPO_ID)
    ap.add_argument("--upload", action="store_true", help="push the re-encoded inputs/ back to HF")
    ap.add_argument("--work-dir", type=Path, default=WORK_DIR, help="where to download / re-encode")
    ap.add_argument("--limit", type=int, help="only process the first N tiles (for a quick check)")
    return ap.parse_args()


def download_inputs(repo_id: str, work_dir: Path) -> Path:
    """Snapshot inputs/ from the dataset repo; returns the local inputs/ directory."""
    print(f"Downloading inputs/ from {repo_id} ...")
    local = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["inputs/*.tif"],
        local_dir=str(work_dir / "download"),
    )
    return Path(local) / "inputs"


def main():
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    src_dir = download_inputs(args.hf_repo, args.work_dir)
    tifs = sorted(src_dir.glob("*.tif"))
    if args.limit:
        tifs = tifs[: args.limit]
    if not tifs:
        sys.exit(f"No input GeoTIFFs found under {src_dir}")

    out_dir = args.work_dir / "inputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    before = after = skipped = 0
    for i, src in enumerate(tifs, 1):
        dest = out_dir / src.name
        with rasterio.open(src) as ds:
            already_encoded = ds.dtypes[0] == "uint16"
        if already_encoded:
            skipped += 1
            continue
        encode_input_for_web(src, dest)
        before += src.stat().st_size
        after  += dest.stat().st_size
        if i % 50 == 0 or i == len(tifs):
            print(f"  {i}/{len(tifs)} tiles")

    mb = 1024 * 1024
    print(f"\nRe-encoded {len(tifs) - skipped} tile(s), skipped {skipped} already-encoded.")
    if before:
        print(f"  {before / mb:.1f} MB -> {after / mb:.1f} MB  ({before / after:.1f}x smaller)")
        print(f"  ~{before / max(len(tifs) - skipped, 1) / mb:.2f} MB -> "
              f"~{after / max(len(tifs) - skipped, 1) / mb:.2f} MB per tile")

    if not args.upload:
        print(f"\nDry run — nothing uploaded. Re-encoded files are in {out_dir}")
        print("Re-run with --upload to publish them.")
        return


    print(f"\nUploading inputs/ to {args.hf_repo} ...")
    HfApi().upload_folder(
        folder_path=str(out_dir),
        path_in_repo="inputs",
        repo_id=args.hf_repo,
        repo_type="dataset",
        commit_message="Re-encode model inputs as uint16 + deflate for faster frontend loading",
    )
    print("  done")


if __name__ == "__main__":
    main()
