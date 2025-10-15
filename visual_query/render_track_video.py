import os
import json
import argparse
import cv2

def load_tracks(pred_json):
    """
    Supports either:
    {
      "video": "...",
      "image": "...",
      "score": 0.95,
      "track": [{"frame": f, "x": x, "y": y, "w": w, "h": h}, ...]
    }
    OR multiple tracks:
    {
      "tracks": [
        {"id": 0, "score": 0.9, "track": [...]},
        {"id": 1, "score": 0.8, "track": [...]}
      ]
    }
    """
    with open(pred_json, "r") as f:
        data = json.load(f)

    tracks = []
    if "track" in data:
        tracks.append({"id": 0, "score": data.get("score", None), "track": data["track"]})
    elif "tracks" in data:
        # normalize ids
        for i, tr in enumerate(data["tracks"]):
            tracks.append({
                "id": tr.get("id", i),
                "score": tr.get("score", None),
                "track": tr["track"]
            })
    else:
        raise ValueError("prediction.json must contain either 'track' or 'tracks'.")

    return tracks, data

def build_frame_index(tracks):
    """
    frame_to_boxes[f] -> list of dicts: {"id": id, "x": x, "y": y, "w": w, "h": h, "score": score}
    """
    frame_to_boxes = {}
    for tr in tracks:
        tid = tr["id"]
        tscore = tr.get("score", None)
        for step in tr["track"]:
            f = int(step["frame"])
            box = {
                "id": tid,
                "x": int(step["x"]),
                "y": int(step["y"]),
                "w": int(step["w"]),
                "h": int(step["h"]),
                "score": tscore
            }
            frame_to_boxes.setdefault(f, []).append(box)
    return frame_to_boxes

def color_for_id(idx):
    # deterministic but distinct-ish colors
    base = [
        (255, 64, 64), (64, 255, 64), (64, 64, 255),
        (255, 192, 64), (192, 64, 255), (64, 255, 192),
        (255, 128, 192), (192, 255, 128), (128, 192, 255)
    ]
    return base[idx % len(base)]

def clamp_box(x, y, w, h, W, H):
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h

def render(args):
    tracks, meta = load_tracks(args.pred)
    frame_to_boxes = build_frame_index(tracks)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    # Get properties
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = args.fps if args.fps is not None and args.fps > 0 else (src_fps if src_fps and src_fps > 0 else 30.0)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer for: {args.out}")

    # Optional top-left overlay text
    title = args.title
    if not title:
        if "video" in meta:
            title = f"Track: {meta.get('video', '')} | score={meta.get('score', '')}"
        else:
            title = "Track Visualization"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = args.font_scale
    thickness = args.thickness

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Draw boxes for this frame
        if frame_idx in frame_to_boxes:
            for box in frame_to_boxes[frame_idx]:
                tid = box["id"]
                color = color_for_id(tid)
                x, y, w, h = clamp_box(box["x"], box["y"], box["w"], box["h"], width, height)
                p1 = (x, y)
                p2 = (x + w, y + h)
                cv2.rectangle(frame, p1, p2, color, thickness)
                label = f"id={tid}"
                if box["score"] is not None:
                    try:
                        label += f" s={float(box['score']):.2f}"
                    except Exception:
                        pass
                cv2.putText(frame, label, (x, max(0, y - 6)), font, font_scale, color, thickness, cv2.LINE_AA)

        # Header info
        cv2.putText(frame, title, (10, 25), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"frame {frame_idx}/{max(nframes-1, frame_idx)}", (10, 50), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Done. Wrote: {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Render tracked boxes onto a video.")
    ap.add_argument("--video", required=True, help="Path to source video used for inference.")
    ap.add_argument("--pred",  required=True, help="Path to prediction.json (single or multiple tracks).")
    ap.add_argument("--out",   default="track_overlay.mp4", help="Output video path (mp4).")
    ap.add_argument("--fps",   type=float, default=None, help="Override output FPS (default: use source).")
    ap.add_argument("--thickness", type=int, default=2, help="Box/label thickness.")
    ap.add_argument("--font-scale", type=float, default=0.7, help="Overlay font scale.")
    ap.add_argument("--title", type=str, default="", help="Custom header text.")
    args = ap.parse_args()
    render(args)
