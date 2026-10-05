import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from run_on_video.run import MomentDETRPredictor, print_diagnostics
from utils.video_utils import validate_video_file


def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS.ss string."""
    seconds = max(0.0, seconds)
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hrs:02d}:{mins:02d}:{secs:05.2f}"


def run_custom(video_path: str, queries: list, ckpt_path: str, top_n: int = 5, min_score: float = 0.0):
    video_path = os.path.abspath(video_path)
    ckpt_path = os.path.abspath(ckpt_path)

    is_valid, err_msg, _ = validate_video_file(video_path, max_duration=150.0)
    if not is_valid:
        print(err_msg, file=sys.stderr)
        return
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint file not found at: {ckpt_path}", file=sys.stderr)
        return

    print_diagnostics()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model_name_or_path = "ViT-B/32"

    predictor = MomentDETRPredictor(
        ckpt_path=ckpt_path,
        clip_model_name_or_path=clip_model_name_or_path,
        device=device
    )

    print("\nRunning Moment-DETR inference on custom video...")
    predictions = predictor.localize_moment(video_path=video_path, query_list=queries)

    print("\n" + "=" * 80)
    print(f"MOMENT RETRIEVAL RESULTS FOR: {os.path.basename(video_path)}")
    print("=" * 80)

    for idx, pred in enumerate(predictions):
        query_str = pred["query"]
        moments = pred["pred_relevant_windows"]
        filtered_moments = [m for m in moments if m[2] >= min_score][:top_n]

        print(f"\n[Query #{idx + 1}]: \"{query_str}\"")
        print("-" * 80)
        if not filtered_moments:
            print("  No moments found above confidence threshold.")
            continue

        print(f"  {'Rank':<5} | {'Start - End (s)':<22} | {'Timestamps (HH:MM:SS)':<25} | {'Confidence':<10}")
        print("  " + "-" * 72)
        for rank, (st, ed, score) in enumerate(filtered_moments, start=1):
            st_clean = max(0.0, st)
            ed_clean = max(0.0, ed)
            time_range_s = f"{st_clean:6.2f}s - {ed_clean:6.2f}s"
            hhmmss_range = f"{format_timestamp(st_clean)} - {format_timestamp(ed_clean)}"
            print(f"  {rank:<5} | {time_range_s:<22} | {hhmmss_range:<25} | {score:<10.4f}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run Moment-DETR temporal video moment retrieval on custom video/query.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file (.mp4, .mkv, .avi, etc.)")
    parser.add_argument("--query", type=str, nargs="+", required=True, help="One or multiple text queries (e.g. --query \"person enters\" \"car stops\")")
    parser.add_argument("--ckpt", type=str, default=str(PROJECT_ROOT / "run_on_video" / "moment_detr_ckpt" / "model_best.ckpt"), help="Path to Moment-DETR checkpoint file")
    parser.add_argument("--top_n", type=int, default=5, help="Top N predicted moments to display (default: 5)")
    parser.add_argument("--min_score", type=float, default=0.0, help="Minimum confidence score threshold (default: 0.0)")

    args = parser.parse_args()
    run_custom(
        video_path=args.video,
        queries=args.query,
        ckpt_path=args.ckpt,
        top_n=args.top_n,
        min_score=args.min_score
    )


if __name__ == "__main__":
    main()
