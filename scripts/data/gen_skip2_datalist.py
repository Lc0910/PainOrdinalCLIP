"""Generate skip-first-2-frames data list from original BioVid data list.

Each BioVid video has 10 frames (sorted by frame index in filename).
This script groups frames by video ID, sorts them, and drops the first 2 frames.

Usage:
    python scripts/data/gen_skip2_datalist.py \
        --input data/biovid/train.txt \
        --output data/biovid/train_skip2.txt

    # Or batch:
    python scripts/data/gen_skip2_datalist.py \
        --input data/biovid/train.txt data/biovid/test.txt \
        --suffix _skip2
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path


def parse_frame_entry(line: str):
    """Parse 'images/071309_w_21-BL1-081_27.jpg 0' into (video_id, frame_idx, full_line).

    video_id = 'images/071309_w_21-BL1-081'
    frame_idx = 27
    """
    parts = line.strip().split()
    if len(parts) < 2:
        return None
    filepath, label = parts[0], parts[1]

    # Extract frame index: last _NNN before .jpg
    match = re.match(r"^(.+)_(\d+)\.jpg$", filepath)
    if not match:
        return None
    video_id = match.group(1)
    frame_idx = int(match.group(2))
    return video_id, frame_idx, line.strip()


def generate_skip2(input_path: Path, output_path: Path, skip_n: int = 2):
    """Read data list, skip first `skip_n` frames per video, write output."""
    videos: dict = defaultdict(list)

    with open(input_path, "r") as f:
        for line in f:
            parsed = parse_frame_entry(line)
            if parsed is None:
                continue
            video_id, frame_idx, full_line = parsed
            videos[video_id].append((frame_idx, full_line))

    # Sort frames within each video by frame index, then drop first `skip_n`
    kept_lines = []
    total_frames = 0
    dropped_frames = 0
    for video_id in sorted(videos.keys()):
        frames = sorted(videos[video_id], key=lambda x: x[0])
        total_frames += len(frames)
        dropped_frames += min(skip_n, len(frames))
        for _, full_line in frames[skip_n:]:
            kept_lines.append(full_line)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for line in kept_lines:
            f.write(line + "\n")

    num_videos = len(videos)
    print(f"  {input_path.name} → {output_path.name}")
    print(f"    Videos: {num_videos}")
    print(f"    Frames: {total_frames} → {len(kept_lines)} (dropped {dropped_frames})")
    print(f"    Frames/video: {total_frames // num_videos} → {len(kept_lines) // num_videos}")


def main():
    parser = argparse.ArgumentParser(description="Generate skip-first-N-frames data list")
    parser.add_argument("--input", "-i", nargs="+", required=True, help="Input data list file(s)")
    parser.add_argument("--output", "-o", nargs="*", default=None, help="Output file(s), auto-generated if omitted")
    parser.add_argument("--suffix", default="_skip2", help="Suffix for auto-generated output names (default: _skip2)")
    parser.add_argument("--skip", type=int, default=2, help="Number of leading frames to skip per video (default: 2)")
    args = parser.parse_args()

    inputs = [Path(p) for p in args.input]
    if args.output:
        outputs = [Path(p) for p in args.output]
        assert len(outputs) == len(inputs), "Number of --output must match --input"
    else:
        outputs = [p.parent / (p.stem + args.suffix + p.suffix) for p in inputs]

    print(f"Skipping first {args.skip} frames per video:\n")
    for inp, out in zip(inputs, outputs):
        generate_skip2(inp, out, skip_n=args.skip)
    print("\nDone.")


if __name__ == "__main__":
    main()
