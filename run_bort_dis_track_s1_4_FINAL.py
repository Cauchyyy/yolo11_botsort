#!/usr/bin/env python3
"""Run BoT-SORT tracking with S1+S4 logic on one or more videos, emitting dual MOT txt outputs."""

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

# Supported video extensions
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".flv"}


def list_videos(input_path: Path) -> List[Path]:
    """Return a sorted list of video files from a file or directory path."""
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        vids = [p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
        vids.sort(key=lambda x: x.name)
        return vids
    return []


# Ensure local repo imports (e.g., ``reid``) are resolvable when running from elsewhere
PROJ_DIR = Path(__file__).resolve().parent
if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))

from ultralytics.trackers.bot_sort import BOTSORT  # noqa: E402


def _load_tracker_cfg(cfg_path: Path) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _merge_tracker_cfg(base: dict, args: argparse.Namespace) -> SimpleNamespace:
    """Merge CLI overrides into the BoT-SORT config YAML and return a namespace."""
    cfg = dict(base)
    overrides = {
        "track_high_thresh": args.track_high_thresh,
        "track_low_thresh": args.track_low_thresh,
        "new_track_thresh": args.new_track_thresh,
        "track_buffer": args.track_buffer,
        "match_thresh": args.match_thresh,
        "fuse_score": args.fuse_score,
        "gmc_method": args.gmc_method,
        "proximity_thresh": args.proximity_thresh,
        "appearance_thresh": args.appearance_thresh,
        "reid_weights": args.reid_weights,
        "reid_device": args.reid_device,
        "stable_frames": args.stable_frames,
        "min_votes": args.min_votes,
        "takeover_hits_margin": args.takeover_hits_margin,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v

    use_reid = bool(args.reid_weights) and args.with_reid
    cfg["with_reid"] = use_reid
    return SimpleNamespace(**cfg)


def _draw_tracks(frame: np.ndarray, tracks: np.ndarray) -> np.ndarray:
    """Draw bounding boxes with display_id on the frame."""
    vis = frame.copy()
    for t in tracks:
        x1, y1, x2, y2, display_id, score, cls_id, idx, perm_id, raw_track_id = t
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        display_id = int(display_id)
        score = float(score)
        cls_id = int(cls_id)
        color = ((display_id * 37) % 255, (display_id * 91) % 255, (display_id * 53) % 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cls_txt = f"f{cls_id}" if cls_id >= 0 else "?"
        label = f"ID:{display_id} {cls_txt} {score:.2f}"
        cv2.putText(vis, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return vis


def _dedup_by_id(tracks: np.ndarray, id_index: int) -> Dict[int, List[float]]:
    """Keep only the highest-score track per ID within a frame."""
    best = {}
    for t in tracks:
        tid = int(t[id_index])
        score = float(t[5])
        if tid not in best or score > best[tid][5]:
            best[tid] = t
    return best


def main():
    parser = argparse.ArgumentParser(description="Run BoT-SORT tracking with S1+S4 and dual MOT exports")
    parser.add_argument("--input", default="/home/waas/mot_data_1107/all_video", help="Input video file or directory")
    parser.add_argument("--out_dir", default="/home/waas/yolo11_track/track_video&txt/TV_S1_4V3", help="Root output directory")
    parser.add_argument("--weights", default="/home/waas/yolo11_track/runs_yolo11n_1209/train/exp/weights/best.pt", help="YOLO weights")
    parser.add_argument("--tracker_cfg", default=str(PROJ_DIR / "ultralytics/cfg/trackers/botsort.yaml"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="cuda")

    # ReID
    parser.add_argument("--with_reid", dest="with_reid", action="store_true", help="Enable ReID")
    parser.add_argument("--no_reid",   dest="with_reid", action="store_false", help="Disable ReID")
    parser.set_defaults(with_reid=True)
    parser.add_argument("--reid_weights", default="/home/waas/ultralytics-yolo11-main/reid_out_ori/reid_best.pth", help="ReID .pt/.pth path")
    parser.add_argument("--reid_device", default="cuda")

    # Tracker thresholds
    parser.add_argument("--track_high_thresh", type=float)
    parser.add_argument("--track_low_thresh", type=float)
    parser.add_argument("--new_track_thresh", type=float)
    parser.add_argument("--track_buffer", type=int)
    parser.add_argument("--track_buffer_perm", type=int, default=60,
                        help="Permanent tracks lost buffer (frames). Default: same as track_buffer")
    parser.add_argument("--match_thresh", type=float)
    parser.add_argument("--fuse_score", action="store_true")
    parser.add_argument("--gmc_method", default=None)
    parser.add_argument("--proximity_thresh", type=float)
    parser.add_argument("--appearance_thresh", type=float)

    # S1/S4 parameters
    parser.add_argument("--stable_frames", type=int, default=5)
    parser.add_argument("--min_votes", type=int, default=3)
    parser.add_argument("--takeover_hits_margin", type=int, default=0)

    args = parser.parse_args()
    print("[CHECK] args.with_reid =", args.with_reid)
    print("[CHECK] args.reid_weights =", args.reid_weights)

    input_path = Path(args.input)
    videos_out = Path(args.out_dir) / "videos"
    txt_display_out = Path(args.out_dir) / "txt" / "display_id"
    txt_track_out = Path(args.out_dir) / "txt" / "track_id"
    videos_out.mkdir(parents=True, exist_ok=True)
    txt_display_out.mkdir(parents=True, exist_ok=True)
    txt_track_out.mkdir(parents=True, exist_ok=True)

    video_list = list_videos(input_path)
    if not video_list:
        print(f"No video found at: {input_path}")
        return 1

    model = YOLO(args.weights)
    print("Loaded YOLO:", args.weights)

    tracker_cfg = _load_tracker_cfg(Path(args.tracker_cfg))
    tracker_args = _merge_tracker_cfg(tracker_cfg, args)
    # B3: allow longer lost buffer for permanent (locked-ID) tracks
    if getattr(args, "track_buffer_perm", None) is not None:
        tracker_args.track_buffer_perm = int(args.track_buffer_perm)
    if tracker_args.with_reid:
        print(f"ReID enabled: {tracker_args.reid_weights} on {tracker_args.reid_device}")
    else:
        print("Running without ReID")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    for vid_path in video_list:
        print(f"\n=== Processing video: {vid_path} ===")
        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            print(f"Cannot open {vid_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        stem = vid_path.stem
        out_video = videos_out / f"{stem}.mp4"
        out_txt_display = txt_display_out / f"{stem}.txt"
        out_txt_track = txt_track_out / f"{stem}.txt"

        writer = cv2.VideoWriter(str(out_video), fourcc, fps, (width, height))
        txt_display = open(out_txt_display, "w")
        txt_track = open(out_txt_track, "w")

        tracker = BOTSORT(args=tracker_args, frame_rate=fps)

        t0 = time.time()
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=args.device, verbose=False)
            boxes = results[0].boxes.cpu().numpy() if len(results) else None

            # BOTSORT.update returns (N, 10): [x1,y1,x2,y2,display_id,score,cls,idx,perm_id,track_id]
            tracks = tracker.update(boxes, frame) if boxes is not None else np.zeros((0, 10), dtype=float)
            if tracks is None:
                tracks = np.zeros((0, 10), dtype=float)

            vis = _draw_tracks(frame, tracks) if len(tracks) else frame
            writer.write(vis)

            if tracks is not None and len(tracks):
                frame_id = frame_idx + 1

                # De-duplicate within frame by ID for display_id and track_id separately
                best_display = _dedup_by_id(tracks, id_index=4)
                best_trackid = _dedup_by_id(tracks, id_index=9)

                # Assert no duplicates remain for display_id
                assert len(best_display) == len(set(best_display.keys())), f"Duplicate display_id at frame {frame_id}: {best_display.keys()}"

                for t in best_display.values():
                    x1, y1, x2, y2, display_id, score, cls_id, idx, perm_id, track_id = t
                    w = float(x2 - x1)
                    h = float(y2 - y1)
                    line = f"{frame_id},{int(display_id)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{float(score):.4f},-1,-1,-1\n"
                    txt_display.write(line)

                for t in best_trackid.values():
                    x1, y1, x2, y2, display_id, score, cls_id, idx, perm_id, track_id = t
                    w = float(x2 - x1)
                    h = float(y2 - y1)
                    line = f"{frame_id},{int(track_id)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{float(score):.4f},-1,-1,-1\n"
                    txt_track.write(line)

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  Processed {frame_idx} frames, elapsed {time.time() - t0:.1f}s")

        cap.release()
        writer.release()
        txt_display.close()
        txt_track.close()
        print(f"Done. Saved: {out_video}, {out_txt_display}, {out_txt_track}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())