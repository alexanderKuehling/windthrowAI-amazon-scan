"""
Summarise the expert reviews collected so far, split by what they can train.

    python scripts/summarize_reviews.py              # full report (downloads the review JSONs)
    python scripts/summarize_reviews.py --fast       # counts only, from the file listing
    python scripts/summarize_reviews.py --since 2026-08-01

Reviews are persisted by the tileloading backend into the results HF dataset as
`reviews/{tile_id}/{ts}__p{changed_pixels}__{reviewer}.json` plus a corrected
mask PNG next to it. Two training signals come out of the same record:

  * **Classification** — the verdict is a tile-level label. Every reviewed tile
    was flagged positive by the classifier, so `confirmed`/`edited` is a
    confirmed true positive and `no_windthrow` is a corrected false positive.
    That is the whole supervision the review round adds to stage 1.
  * **Segmentation** — the mask PNG is a pixel-level target. `edited` is a
    genuinely corrected mask, `confirmed` endorses the proposed one unchanged,
    `no_windthrow` implies an empty mask.

The report answers how many *distinct* tiles carry each signal and how many
reviews per tile there are on average — i.e. whether there is enough to retrain
and how much reviewer redundancy exists per image.

`--fast` skips the JSON download and works off the filenames alone (they encode
timestamp, changed-pixel count and reviewer, see ranking._parse_stem in the
tileloading backend). That is enough for counts and per-tile averages, but not
for the verdicts — so the classification/segmentation split needs a full run.

Two CSVs are written alongside the report (`--no-csv` to suppress): one row per
reviewed tile and one per reviewer. The per-tile file is what the retraining
step reads to build its label set.

Requires: huggingface_hub. Read access is public; export HF_TOKEN (or
`huggingface-cli login`) if the dataset is ever made private.
"""

import argparse
import csv
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Same dataset the scan exports to (scripts/export_for_frontend.py). Declared
# here rather than imported so this script stays free of the GDAL stack.
HF_REPO_ID = "alexKlg/windthrowAI-dataset-segmentation_results"

WORK_DIR = ROOT / "outputs_final" / "reviews"

# Verdict vocabulary of the review backend (tileloading reviews.py).
POSITIVE_VERDICTS = {"confirmed", "edited"}   # tile does show windthrow
NEGATIVE_VERDICTS = {"no_windthrow"}          # classifier false positive
CORRECTED_VERDICT = "edited"                  # mask was actually redrawn
VERDICT_ORDER     = ["confirmed", "edited", "no_windthrow"]

TS_FORMAT = "%Y%m%dT%H%M%S%fZ"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-repo", default=HF_REPO_ID)
    ap.add_argument("--fast", action="store_true",
                    help="skip the JSON download; counts only, no verdict breakdown")
    ap.add_argument("--since", metavar="YYYY-MM-DD",
                    help="only count reviews submitted on or after this date")
    ap.add_argument("--work-dir", type=Path, default=WORK_DIR,
                    help="where the review JSONs are downloaded to")
    ap.add_argument("--out-dir", type=Path, default=WORK_DIR,
                    help="where the per-tile / per-reviewer CSVs are written")
    ap.add_argument("--no-csv", action="store_true", help="report only, write no CSVs")
    ap.add_argument("--top", type=int, default=10, help="how many reviewers to list")
    return ap.parse_args()


# ── Loading ──────────────────────────────────────────────────────────────

def parse_stem(stem: str) -> tuple:
    """(timestamp, changed_pixels, reviewer) from a review filename stem.

    New format    "<ts>__p<pixels>__<reviewer>", legacy "<ts>__<reviewer>"
    (pixel count unknown -> 0). The reviewer may itself contain "__", so only
    the fixed-shape leading segments are peeled off — mirrors _parse_stem in
    the tileloading backend's ranking.py.
    """
    parts = stem.split("__", 2)
    if len(parts) < 2:
        return stem, 0, ""
    if len(parts) == 3 and len(parts[1]) > 1 and parts[1][0] == "p" and parts[1][1:].isdigit():
        return parts[0], int(parts[1][1:]), parts[2]
    return parts[0], 0, stem.split("__", 1)[1]


def list_review_files(repo_id: str, token) -> list:
    """Every review JSON in the dataset repo, as repo-relative paths."""
    from huggingface_hub import HfApi

    files = HfApi(token=token).list_repo_files(repo_id=repo_id, repo_type="dataset")
    return sorted(f for f in files if f.startswith("reviews/") and f.endswith(".json"))


def records_from_listing(paths: list) -> list:
    """Build records from the filenames alone (--fast): no verdicts."""
    records = []
    for path in paths:
        parts = path.split("/")
        if len(parts) != 3:
            continue  # not reviews/{tile_id}/{stem}.json
        _, tile_id, name = parts
        ts, changed, reviewer = parse_stem(name[: -len(".json")])
        records.append({
            "tile_id": tile_id, "verdict": None, "reviewer": reviewer,
            "changed_pixels": changed, "width": 0, "height": 0,
            "timestamp": ts, "mask_file": None, "path": path,
        })
    return records


def records_from_json(repo_id: str, token, work_dir: Path, paths: list) -> list:
    """Download every review JSON and read the full record, verdict included.

    The JSONs are a few hundred bytes each and the snapshot is incremental, so
    a re-run only fetches reviews added since the last one.
    """
    from huggingface_hub import snapshot_download

    print(f"Downloading {len(paths)} review record(s) from {repo_id} ...")
    local = Path(snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["reviews/*/*.json"],
        local_dir=str(work_dir / "download"),
        token=token,
    ))

    records, unreadable = [], 0
    for path in paths:
        parts = path.split("/")
        if len(parts) != 3:
            continue
        _, tile_id, name = parts
        ts, changed, reviewer = parse_stem(name[: -len(".json")])
        try:
            with open(local / path) as f:
                rec = json.load(f)
        except (OSError, json.JSONDecodeError):
            unreadable += 1
            rec = {}
        # The filename is the authoritative fallback: it is what /rankings
        # aggregates on, and it survives records written by older backends.
        records.append({
            "tile_id":        rec.get("tile_id") or tile_id,
            "verdict":        rec.get("verdict"),
            "reviewer":       rec.get("reviewer") or reviewer,
            "changed_pixels": int(rec.get("changed_pixels") or changed),
            "width":          int(rec.get("width") or 0),
            "height":         int(rec.get("height") or 0),
            "timestamp":      rec.get("timestamp") or ts,
            "mask_file":      rec.get("mask_file"),
            "path":           path,
        })
    if unreadable:
        print(f"  !! {unreadable} record(s) could not be read — counted without a verdict")
    return records


def published_positive_count(repo_id: str, token) -> int:
    """How many tiles are up for review at all (rows in inputs_index.csv).

    Returns 0 when the index cannot be fetched — the report then just omits the
    progress line instead of failing.
    """
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(repo_id=repo_id, repo_type="dataset",
                               filename="inputs_index.csv", token=token)
        with open(path, newline="") as f:
            return sum(1 for _ in csv.DictReader(f))
    except Exception as exc:  # noqa: BLE001 — the index is a nice-to-have here
        print(f"  (inputs_index.csv unavailable: {exc})")
        return 0


def parse_ts(ts: str):
    try:
        return datetime.strptime(ts, TS_FORMAT).replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


# ── Reporting ────────────────────────────────────────────────────────────

def pct(n: int, total: int) -> str:
    return f"{100.0 * n / total:5.1f}%" if total else "    –"


def per_tile_stats(by_tile: dict) -> tuple:
    """(mean, median, max) reviews per tile."""
    counts = [len(v) for v in by_tile.values()]
    if not counts:
        return 0.0, 0, 0
    return statistics.mean(counts), int(statistics.median(counts)), max(counts)


def print_overall(records: list, by_tile: dict, published: int, top: int):
    print("\nOverall")
    print(f"  Reviews              : {len(records)}")
    print(f"  Distinct tiles       : {len(by_tile)}")
    mean, median, largest = per_tile_stats(by_tile)
    print(f"  Reviews per tile     : {mean:.2f} mean · {median} median · {largest} max")

    hist = Counter(len(v) for v in by_tile.values())
    for n in sorted(hist):
        label = f"{n} review" + ("s" if n != 1 else "")
        print(f"    {label:<18}: {hist[n]:5d} tiles  ({pct(hist[n], len(by_tile))})")

    reviewers = Counter(r["reviewer"] or "(anonymous)" for r in records)
    print(f"  Reviewers            : {len(reviewers)}")
    for name, n in reviewers.most_common(top):
        tiles = len({r['tile_id'] for r in records if (r['reviewer'] or '(anonymous)') == name})
        print(f"    {name:<18}: {n:5d} reviews on {tiles} tiles")
    if len(reviewers) > top:
        print(f"    ... and {len(reviewers) - top} more")

    stamps = sorted(t for t in (parse_ts(r["timestamp"]) for r in records) if t)
    if stamps:
        week = datetime.now(timezone.utc) - timedelta(days=7)
        recent = sum(1 for t in stamps if t >= week)
        print(f"  First / last review  : {stamps[0]:%Y-%m-%d} · {stamps[-1]:%Y-%m-%d}"
              f"   ({recent} in the last 7 days)")

    if published:
        print(f"  Review progress      : {len(by_tile)} / {published} published positives "
              f"({pct(len(by_tile), published).strip()}) — {published - len(by_tile)} untouched")


def majority_label(verdicts: list) -> str:
    """'windthrow' / 'no_windthrow' / 'tied' from one tile's verdicts."""
    pos = sum(1 for v in verdicts if v in POSITIVE_VERDICTS)
    neg = sum(1 for v in verdicts if v in NEGATIVE_VERDICTS)
    if pos == neg:
        return "tied"
    return "windthrow" if pos > neg else "no_windthrow"


def print_classification(records: list, by_tile: dict):
    print("\nClassification  —  tile-level label from the verdict")
    labelled = {t: [r for r in rs if r["verdict"]] for t, rs in by_tile.items()}
    labelled = {t: rs for t, rs in labelled.items() if rs}
    n_labels = sum(len(rs) for rs in labelled.values())
    if not labelled:
        print("  No verdicts available — run without --fast.")
        return

    mean, median, largest = per_tile_stats(labelled)
    print(f"  Labelled tiles       : {len(labelled)}")
    print(f"  Labels               : {n_labels}  ({mean:.2f} per tile, {median} median, {largest} max)")

    verdicts = Counter(r["verdict"] for r in records if r["verdict"])
    parts = [f"{v} {verdicts.get(v, 0)}" for v in VERDICT_ORDER if verdicts.get(v)]
    other = {v: n for v, n in verdicts.items() if v not in VERDICT_ORDER}
    parts += [f"{v} {n}" for v, n in sorted(other.items())]
    print(f"  Verdicts             : {' · '.join(parts)}")

    labels = Counter(majority_label([r["verdict"] for r in rs]) for rs in labelled.values())
    print(f"  Majority label       : windthrow {labels.get('windthrow', 0)} tiles · "
          f"no windthrow {labels.get('no_windthrow', 0)} · tied {labels.get('tied', 0)}")

    multi = {t: rs for t, rs in labelled.items() if len(rs) > 1}
    if multi:
        unanimous = sum(1 for rs in multi.values()
                        if len({r["verdict"] in POSITIVE_VERDICTS for r in rs}) == 1)
        print(f"  Multi-reviewed tiles : {len(multi)} — {unanimous} unanimous "
              f"({pct(unanimous, len(multi)).strip()}), {len(multi) - unanimous} split")
    else:
        print("  Multi-reviewed tiles : none — every tile has a single review")

    # Every reviewed tile passed the stage-1 gate, so the label set the round
    # produces is exactly: confirmed true positives + corrected false positives.
    print(f"  → training signal    : {labels.get('windthrow', 0)} positive / "
          f"{labels.get('no_windthrow', 0)} negative tiles "
          f"(all of them tiles the current classifier called positive)")


def print_segmentation(records: list, by_tile: dict):
    print("\nSegmentation  —  corrected mask per review")
    with_verdict = [r for r in records if r["verdict"]]
    if not with_verdict:
        with_mask = [r for r in records if r["changed_pixels"] > 0]
        print("  Reviews with a mask  : unknown without verdicts — run without --fast")
        print(f"  Reviews that changed pixels: {len(with_mask)} "
              f"on {len({r['tile_id'] for r in with_mask})} tiles")
        return

    edited    = [r for r in with_verdict if r["verdict"] == CORRECTED_VERDICT]
    usable    = [r for r in with_verdict if r["verdict"] in POSITIVE_VERDICTS]
    empty     = [r for r in with_verdict if r["verdict"] in NEGATIVE_VERDICTS]

    edited_tiles = {r["tile_id"] for r in edited}
    usable_tiles = {r["tile_id"] for r in usable}

    print(f"  Corrected masks      : {len(edited)} reviews on {len(edited_tiles)} distinct tiles")
    print(f"  Confirmed as-is      : {len(usable) - len(edited)} reviews "
          f"(proposed mask endorsed unchanged)")
    print(f"  Empty-mask targets   : {len(empty)} reviews "
          f"(no_windthrow — background-only target)")

    by_usable = defaultdict(list)
    for r in usable:
        by_usable[r["tile_id"]].append(r)
    mean, median, largest = per_tile_stats(by_usable)
    print(f"  Usable mask targets  : {len(usable)} on {len(usable_tiles)} tiles "
          f"({mean:.2f} per tile, {median} median, {largest} max)")

    if edited:
        changed = [r["changed_pixels"] for r in edited]
        print(f"  Pixels changed       : {statistics.mean(changed):8.0f} mean · "
              f"{statistics.median(changed):8.0f} median · {max(changed):8d} max")
        fracs = [100.0 * r["changed_pixels"] / (r["width"] * r["height"])
                 for r in edited if r["width"] and r["height"]]
        if fracs:
            print(f"  Share of tile edited : {statistics.mean(fracs):.2f}% mean · "
                  f"{statistics.median(fracs):.2f}% median")

    multi = {t: rs for t, rs in by_usable.items() if len(rs) > 1}
    print(f"  Tiles with 2+ masks  : {len(multi)}"
          + ("  (reviewer agreement per pixel is not computed here)" if multi else ""))


# ── CSV output ───────────────────────────────────────────────────────────

def write_tile_csv(path: Path, by_tile: dict):
    """One row per reviewed tile — the label set the retraining step reads."""
    cols = ["tile_id", "reviews", "reviewers", "confirmed", "edited", "no_windthrow",
            "majority_label", "unanimous", "changed_pixels_mean", "changed_pct_mean",
            "first_review", "last_review", "latest_mask"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for tile_id in sorted(by_tile):
            rs = by_tile[tile_id]
            verdicts = [r["verdict"] for r in rs if r["verdict"]]
            counts = Counter(verdicts)
            fracs = [100.0 * r["changed_pixels"] / (r["width"] * r["height"])
                     for r in rs if r["width"] and r["height"]]
            stamps = sorted(s for s in (parse_ts(r["timestamp"]) for r in rs) if s)
            masked = [r for r in sorted(rs, key=lambda r: r["timestamp"]) if r["mask_file"]]
            w.writerow({
                "tile_id":             tile_id,
                "reviews":             len(rs),
                "reviewers":           len({r["reviewer"] for r in rs if r["reviewer"]}),
                "confirmed":           counts.get("confirmed", 0),
                "edited":              counts.get("edited", 0),
                "no_windthrow":        counts.get("no_windthrow", 0),
                "majority_label":      majority_label(verdicts) if verdicts else "",
                "unanimous":           int(len({v in POSITIVE_VERDICTS for v in verdicts}) == 1)
                                       if verdicts else "",
                "changed_pixels_mean": round(statistics.mean([r["changed_pixels"] for r in rs]), 1),
                "changed_pct_mean":    round(statistics.mean(fracs), 4) if fracs else "",
                "first_review":        f"{stamps[0]:%Y-%m-%d}" if stamps else "",
                "last_review":         f"{stamps[-1]:%Y-%m-%d}" if stamps else "",
                "latest_mask":         masked[-1]["mask_file"] if masked else "",
            })
    print(f"  wrote {path}  ({len(by_tile)} tiles)")


def write_reviewer_csv(path: Path, records: list):
    cols = ["reviewer", "reviews", "tiles", "confirmed", "edited", "no_windthrow",
            "changed_pixels_total"]
    by_reviewer = defaultdict(list)
    for r in records:
        by_reviewer[r["reviewer"] or "(anonymous)"].append(r)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for name, rs in sorted(by_reviewer.items(), key=lambda kv: -len(kv[1])):
            counts = Counter(r["verdict"] for r in rs if r["verdict"])
            w.writerow({
                "reviewer":             name,
                "reviews":              len(rs),
                "tiles":                len({r["tile_id"] for r in rs}),
                "confirmed":            counts.get("confirmed", 0),
                "edited":               counts.get("edited", 0),
                "no_windthrow":         counts.get("no_windthrow", 0),
                "changed_pixels_total": sum(r["changed_pixels"] for r in rs),
            })
    print(f"  wrote {path}  ({len(by_reviewer)} reviewers)")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_WRITE_TOKEN")

    paths = list_review_files(args.hf_repo, token)
    if not paths:
        sys.exit(f"No reviews found in {args.hf_repo} (looked for reviews/*/*.json).")

    if args.fast:
        records = records_from_listing(paths)
    else:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        records = records_from_json(args.hf_repo, token, args.work_dir, paths)

    if args.since:
        cutoff = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        before = len(records)
        records = [r for r in records if (parse_ts(r["timestamp"]) or cutoff) >= cutoff]
        print(f"Filtered to reviews since {args.since}: {len(records)} of {before}")
    if not records:
        sys.exit("No reviews left after filtering.")

    by_tile = defaultdict(list)
    for r in records:
        by_tile[r["tile_id"]].append(r)

    published = published_positive_count(args.hf_repo, token)

    header = f"Expert reviews — {args.hf_repo}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    print_overall(records, by_tile, published, args.top)
    print_classification(records, by_tile)
    print_segmentation(records, by_tile)

    if args.fast:
        print("\n(--fast: verdicts were not read; re-run without it for the full split.)")

    if not args.no_csv:
        print("")
        write_tile_csv(args.out_dir / "review_summary_per_tile.csv", by_tile)
        write_reviewer_csv(args.out_dir / "review_summary_per_reviewer.csv", records)
    print("")


if __name__ == "__main__":
    main()
