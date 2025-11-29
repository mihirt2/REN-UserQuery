import os
import json
import argparse
import cv2

def _to_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

def color_for_id(idx):
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

def load_tracks(pred_json):
    with open(pred_json, "r") as f:
        data = json.load(f)

    tracks = []
    if "track" in data:
        tracks.append({"id": 0, "score": data.get("score", None), "track": data["track"]})
    elif "tracks" in data:
        for i, tr in enumerate(data["tracks"]):
            tracks.append({
                "id": tr.get("id", i),
                "score": tr.get("score", None),
                "track": tr["track"]
            })
    else:
        raise ValueError("prediction.json must contain either 'track' or 'tracks'.")

    # optional per-frame scores
    frame_scores_map = {}
    if "frame_scores" in data and isinstance(data["frame_scores"], list):
        for fs in data["frame_scores"]:
            try:
                f = int(fs.get("frame", 0))
                s = float(fs.get("score", None))
                frame_scores_map[f] = s
            except Exception:
                pass

    top_query_points = data.get("query_points", [])
    return tracks, data, top_query_points, frame_scores_map

def build_frame_index(tracks, frame_scores_map):
    frame_to_boxes = {}
    frame_to_embedded_points = {}

    for tr in tracks:
        tid = tr["id"]
        tscore = tr.get("score", None)  # track-level fallback
        for step in tr["track"]:
            f = _to_int(step.get("frame", 0))
            # prefer step score, then frame score, else track score
            step_score = step.get("score", None)
            if step_score is not None:
                try:
                    used_score = float(step_score)
                except Exception:
                    used_score = tscore
            else:
                used_score = frame_scores_map.get(f, tscore)

            box = {
                "id": tid,
                "x": _to_int(step.get("x", 0)),
                "y": _to_int(step.get("y", 0)),
                "w": _to_int(step.get("w", 1)),
                "h": _to_int(step.get("h", 1)),
                "score": used_score,
            }
            frame_to_boxes.setdefault(f, []).append(box)

            if "x2" in step and "y2" in step:
                qpt = {"x": _to_int(step["x2"]), "y": _to_int(step["y2"])}
                frame_to_embedded_points.setdefault(f, []).append(qpt)

    return frame_to_boxes, frame_to_embedded_points

def build_query_points_index(query_points):
    frame_to_points = {}
    global_points = []   # keep empty so we don't draw any global/orange
    for pt in query_points or []:
        f = _to_int(pt.get("frame", 0))
        x = _to_int(pt.get("x", 0))
        y = _to_int(pt.get("y", 0))
        frame_to_points.setdefault(f, []).append({"x": x, "y": y})
    return frame_to_points, global_points

def point_in_any_box(px, py, boxes_xywh):
    """Return True iff (px, py) is inside at least one (x, y, w, h) in boxes_xywh."""
    for (x, y, w, h) in boxes_xywh:
        if x <= px <= x + w and y <= py <= y + h:
            return True
    return False

def render(args):
    QUERY_BLUE = (255, 0, 0)
    tracks, meta, top_query_points, frame_scores_map = load_tracks(args.pred)
    frame_to_boxes, frame_to_embedded_points = build_frame_index(tracks, frame_scores_map)
    frame_to_top_points, global_query_points = build_query_points_index(top_query_points)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = args.fps if args.fps and args.fps > 0 else (src_fps if src_fps and src_fps > 0 else 30.0)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer for: {args.out}")

    title = args.title or (f"Track: {meta.get('video', '')}"
                           if "video" in meta else "Track Visualization")

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = args.font_scale
    thickness = args.thickness

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # List of clamped boxes for this frame, for hit-testing query points
        boxes_xywh = []

        # ----------------- DRAW TRACK BOXES -----------------
        if frame_idx in frame_to_boxes:
            for box in frame_to_boxes[frame_idx]:
                tid = box["id"]
                color = color_for_id(tid)
                x, y, w, h = clamp_box(box["x"], box["y"], box["w"], box["h"], width, height)
                boxes_xywh.append((x, y, w, h))

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

                label = f"id={tid}"
                if box["score"] is not None:
                    try:
                        label += f" s={float(box['score']):.2f}"
                    except Exception:
                        pass
                cv2.putText(frame, label, (x, max(0, y - 6)), font, font_scale,
                            color, thickness, cv2.LINE_AA)

                # Box anchor point
                cv2.circle(frame, (x, y), 5, color, thickness=-1, lineType=cv2.LINE_AA)

        # ----------------- DRAW QUERY POINTS ONLY IF THEY BELONG TO A TRACK -----------------
        if boxes_xywh:
            # global_query_points (usually empty, but keep consistent)
            for pt in global_query_points:
                px, py = pt["x"], pt["y"]
                px = max(0, min(px, width - 1))
                py = max(0, min(py, height - 1))
                if point_in_any_box(px, py, boxes_xywh):
                    cv2.circle(frame, (px, py), 12, (255, 255, 255), thickness=3,
                               lineType=cv2.LINE_AA)
                    cv2.circle(frame, (px, py), 8, QUERY_BLUE, thickness=-1,
                               lineType=cv2.LINE_AA)
                    cv2.drawMarker(frame, (px, py), QUERY_BLUE,
                                   markerType=cv2.MARKER_CROSS,
                                   markerSize=20, thickness=2,
                                   line_type=cv2.LINE_AA)

            # Per-frame REN query point(s) – only if they fall inside some box
            if frame_idx in frame_to_top_points:
                for pt in frame_to_top_points[frame_idx]:
                    px, py = pt["x"], pt["y"]
                    px = max(0, min(px, width - 1))
                    py = max(0, min(py, height - 1))
                    if point_in_any_box(px, py, boxes_xywh):
                        cv2.circle(frame, (px, py), 12, (255, 255, 255), thickness=3,
                                   lineType=cv2.LINE_AA)
                        cv2.circle(frame, (px, py), 8, QUERY_BLUE, thickness=-1,
                                   lineType=cv2.LINE_AA)
                        cv2.drawMarker(frame, (px, py), QUERY_BLUE,
                                       markerType=cv2.MARKER_CROSS,
                                       markerSize=20, thickness=2,
                                       line_type=cv2.LINE_AA)

        # ----------------- HEADER (always white) + CURRENT FRAME SCORE -----------------
        y0, y1, y2 = 25, 50, 75
        cv2.putText(frame, title, (10, y0), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)

        cur_score = None
        if frame_idx in frame_scores_map:
            cur_score = frame_scores_map[frame_idx]
        elif frame_idx in frame_to_boxes:
            try:
                cur_score = max(
                    [b["score"] for b in frame_to_boxes[frame_idx]
                     if b["score"] is not None],
                    default=None
                )
            except Exception:
                cur_score = None

        if cur_score is not None:
            cv2.putText(frame, f"score {cur_score:.4f}", (10, y1), font, font_scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)
        else:
            cv2.putText(frame, "score n/a", (10, y1), font, font_scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)

        cv2.putText(frame, f"frame {frame_idx}/{max(nframes-1, frame_idx)}",
                    (10, y2), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Done. Wrote: {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Render tracked boxes and query points onto a video.")
    ap.add_argument("--video", required=True, help="Path to source video used for inference.")
    ap.add_argument("--pred",  required=True, help="Path to prediction.json (single or multiple tracks).")
    ap.add_argument("--out",   default="track_overlay.mp4", help="Output video path (mp4).")
    ap.add_argument("--fps",   type=float, default=None, help="Override output FPS (default: use source).")
    ap.add_argument("--thickness", type=int, default=2, help="Box/label thickness.")
    ap.add_argument("--font-scale", type=float, default=0.7, help="Overlay font scale.")
    ap.add_argument("--title", type=str, default="", help="Custom header text.")
    args = ap.parse_args()
    render(args)
