#!/usr/bin/env python3
"""Run BoT-SORT tracking (S1+S4 ablation) with ReID fish labels on one or more videos."""

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

# 支持的视频后缀
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".flv"}


def list_videos(input_path: Path):
    """如果是文件就返回单个列表，如果是目录则返回该目录下所有视频文件（按名字排序）"""
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

from ultralytics.trackers.bot_sort import BOTSORT


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
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v

    use_reid = bool(args.reid_weights) and not args.no_reid
    cfg["with_reid"] = use_reid
    return SimpleNamespace(**cfg)


def _draw_tracks(frame: np.ndarray, tracks: np.ndarray) -> np.ndarray:
    vis = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = map(int, t[:4])
        display_id = int(t[4])  # 5th column: per-frame display ID (unique within frame)
        score = float(t[5])  # detection score
        cls_id = int(t[6])  # ReID / class ID
        perm_id = int(t[8]) if len(t) > 8 else -1  # perm_id (1..9) if available
        color = ((display_id * 37) % 255, (display_id * 91) % 255, (display_id * 53) % 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cls_txt = f"f{cls_id}" if cls_id >= 0 else "?"
        label = f"ID:{display_id} {cls_txt} {score:.2f}"
        cv2.putText(vis, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return vis


def main():
    parser = argparse.ArgumentParser(description="Run BoT-SORT tracking with ReID fish IDs (S1+S4 ablation)")
    # 可以传入单个视频文件，也可以传入一个目录（会遍历该目录下所有视频）
    parser.add_argument("--video", default="/home/waas/mot_data_1107/all_video", help="Path to input video file or directory")
    # 输出目录：每个视频各自生成一个 mp4 / txt / npy
    parser.add_argument("--out_dir", default="/home/waas/yolo11_track/track_video&txt/s1/_8_4", help="Directory for output videos")
    parser.add_argument("--npy_dir", default="/home/waas/yolo11_track/track_video&txt/s1/_8_4", help="Directory for numpy outputs")
    parser.add_argument("--txt_dir", default="/home/waas/yolo11_track/track_video&txt/s1/_8_4", help="Directory for MOT-style txt outputs")
    parser.add_argument("--weights", default="/home/waas/mot_data_1107/all_video", help="YOLO weights")
    parser.add_argument("--tracker_cfg", default=str(PROJ_DIR / "ultralytics/cfg/trackers/botsort.yaml"))
    parser.add_argument("--conf_thresh", type=float, default=0.25)

    parser.add_argument("--reid_weights", default="/home/waas/ultralytics-yolo11-main/reid_out_ori/reid_best.pth", help="ReID .pt/.pth path")
    parser.add_argument("--reid_device", default="cuda", help="device for ReID model")
    parser.add_argument("--no_reid", action="store_true", help="disable ReID even if weights are provided")

    parser.add_argument("--track_high_thresh", type=float)
    parser.add_argument("--track_low_thresh", type=float)
    parser.add_argument("--new_track_thresh", type=float)
    parser.add_argument("--track_buffer", type=int)
    parser.add_argument("--match_thresh", type=float)
    parser.add_argument("--fuse_score", action="store_true")
    parser.add_argument("--gmc_method", default=None)
    parser.add_argument("--proximity_thresh", type=float)
    parser.add_argument("--appearance_thresh", type=float)

    args = parser.parse_args()

    video_root = Path(args.video)
    out_dir = Path(args.out_dir)
    txt_dir = Path(args.txt_dir)
    npy_dir = Path(args.npy_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)

    video_list = list_videos(video_root)
    if not video_list:
        print(f"No video found at: {video_root}")
        return 1

    model = YOLO(args.weights)
    print("Loaded YOLO:", args.weights)

    tracker_cfg = _load_tracker_cfg(Path(args.tracker_cfg))
    tracker_args = _merge_tracker_cfg(tracker_cfg, args)
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
        out_video = out_dir / f"{stem}.mp4"
        out_txt = txt_dir / f"{stem}.txt"
        out_npy = npy_dir / f"{stem}.npy"

        writer = cv2.VideoWriter(str(out_video), fourcc, fps, (width, height))
        txt_f = open(out_txt, "w")

        tracker = BOTSORT(args=tracker_args, frame_rate=fps)  # 每个视频重置一个 tracker

        outputs_per_frame = []
        t0 = time.time()
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, conf=args.conf_thresh, verbose=False)
            boxes = results[0].boxes.cpu().numpy() if len(results) else None

            # BOTSORT.update 返回形状为 (N, 10)：[x1,y1,x2,y2,display_id,score,cls,idx,perm_id,track_id]
            tracks = tracker.update(boxes, frame) if boxes is not None else np.zeros((0, 10), dtype=float)
            if tracks is None:
                tracks = np.zeros((0, 10), dtype=float)

            vis = _draw_tracks(frame, tracks) if len(tracks) else frame
            writer.write(vis)
            outputs_per_frame.append(tracks)

            # 写 MOT 格式 txt：frame,id,x,y,w,h,1,1,score（使用原始 track_id 便于 TrackEval）
            if tracks is not None and len(tracks):
                frame_id = frame_idx + 1  # MOT 帧编号从 1 开始
                for t in tracks:
                    x1, y1, x2, y2 = map(float, t[:4])
                    track_id = int(t[9])  # 原始 track_id
                    score = float(t[5])
                    w = x2 - x1
                    h = y2 - y1
                    line = f"{frame_id},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,1,{score:.4f}\n"
                    txt_f.write(line)

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  Processed {frame_idx} frames, elapsed {time.time() - t0:.1f}s")

        cap.release()
        writer.release()
        txt_f.close()
        np.save(out_npy, np.array(outputs_per_frame, dtype=object))
        print(f"Done. Saved: {out_video}, {out_txt}, {out_npy}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())