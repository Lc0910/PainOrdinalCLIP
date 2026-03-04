"""Parse frame-level and video-level stats into a single CSV.

Usage:
    python scripts/experiments/parse_video_results.py -d results/
    python scripts/experiments/parse_video_results.py -d results/ -o results/stats/
"""
import argparse
import datetime
import json
import re
from pathlib import Path

import pandas as pd


def prepare_exp_name(stats_dict):
    """Extract experiment name from output_dir / ckpt_path (same logic as parse_results.py)."""
    exp_name = []

    output_dir = stats_dict.pop("output_dir", None)
    if output_dir is not None:
        output_dir = Path(output_dir)
        if output_dir.stem.startswith("version_"):
            exp_name += [output_dir.parent.stem, output_dir.stem.replace("_", "-")]
        else:
            exp_name.append(output_dir.stem)

    ckpt_name = stats_dict.pop("ckpt_path", None)
    if ckpt_name is not None and ckpt_name != "":
        ckpt_name = Path(ckpt_name)
        exp_name.append(ckpt_name.stem.replace("_", "-"))

    exp_name = "_".join(exp_name)
    try:
        head, tail = re.split("fold-[0-9]+", exp_name, 1)
        mid = re.findall("fold-[0-9]+", exp_name)[0]
        if len(head) > 0 and head[-1] == "-":
            head = head[:-1] + "_"
        if len(tail) > 0 and tail[0] == "-":
            tail = "_" + tail[1:]
        exp_name = head + mid + tail
    except (IndexError, ValueError):
        pass

    stats_dict["exp_name"] = exp_name


def parse_one_stats_file(stats_path, level):
    """Parse a single stats JSON file (one JSON per line)."""
    stats_path = Path(stats_path)
    if not stats_path.is_file():
        return None

    rows = []
    with open(stats_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stats_dict = json.loads(line)
            prepare_exp_name(stats_dict)
            stats_dict["level"] = level
            rows.append(stats_dict)

    if not rows:
        return None
    return pd.DataFrame(rows)


def parse_results_dir(results_dir, run_type="test"):
    """Recursively find frame and video stats files, parse into DataFrame."""
    results_dir = Path(results_dir)
    dfs = []

    # Frame-level: {run_type}_stats.json
    for f in sorted(results_dir.rglob(f"{run_type}_stats.json")):
        df = parse_one_stats_file(f, level="frame")
        if df is not None:
            dfs.append(df)

    # Video-level: {run_type}_video_stats.json
    for f in sorted(results_dir.rglob(f"{run_type}_video_stats.json")):
        df = parse_one_stats_file(f, level="video")
        if df is not None:
            dfs.append(df)

    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse frame + video stats into CSV")
    parser.add_argument("--results_dir", "-d", type=str, default="results/")
    parser.add_argument("--output_dir", "-o", type=str, default="results/stats/")
    parser.add_argument("--output_name", "-n", type=str, default=None)
    parser.add_argument("--run_type", "-r", type=str, default="test", choices=("test", "val"))
    parser.add_argument("--filter", "-f", type=str, default=None,
                        help="Only include experiments whose exp_name contains this substring")
    args = parser.parse_args()

    print(f"Scanning: {args.results_dir}")
    stats = parse_results_dir(args.results_dir, run_type=args.run_type)

    if stats is None or stats.empty:
        print(f"No {args.run_type}_stats.json or {args.run_type}_video_stats.json found under {args.results_dir}")
        exit(1)

    if args.filter and "exp_name" in stats.columns:
        stats = stats[stats["exp_name"].str.contains(args.filter, case=False, na=False)]
        if stats.empty:
            print(f"No experiments matching filter '{args.filter}'")
            exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output_name is None:
        args.output_name = Path(args.results_dir).name

    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    file_path = output_dir / f"{args.output_name}_video.{now}.csv"
    stats.to_csv(str(file_path), index=False)
    print(f"Written to: {file_path}")
    print(stats.to_string(index=False))
