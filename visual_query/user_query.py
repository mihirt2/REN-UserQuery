import os
import json
import yaml
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2

from ren import REN
from visual_query.models import VisualQueryTracker
from visual_query.vq_utils import crop_using_bbox, generate_token_from_bbox

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def read_video_rgb(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"No frames found in: {path}")
    arr = np.stack(frames).astype(np.uint8)
    print("[read_video_rgb] video shape:", arr.shape)  # [T, H, W, 3]
    return arr


def read_image_rgb(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    out = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
    print("[read_image_rgb] image shape:", out.shape)  # [H, W, 3]
    return out


class QueryEncoder:
    def __init__(self, config):
        self.ren = REN(config)
        self.cropping_factors = config['visual_query']['query_cropping_factors']
        self.sam_ckpt = config['data']['sam2_ckpt']
        self.sam_cfg = config['visual_query']['tracker_param']

    def encode_image(self, image, bbox=None):
        """
        Encode a query image into tokens.
        bbox is [x1, y1, x2, y2] in image coordinates.
        If bbox is None, use the full image.
        """
        h, w = image.shape[:2]
        if bbox is None:
            bbox = [0, 0, w, h]
        else:
            # clip bbox to be within image
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(x1 + 1, min(w, x2))
            y2 = max(y1 + 1, min(h, y2))
            bbox = [x1, y1, x2, y2]

        print("[QueryEncoder.encode_image] image shape:", image.shape)
        print("[QueryEncoder.encode_image] bbox:", bbox)
        print("[QueryEncoder.encode_image] cropping_factors:", self.cropping_factors)

        crops, boxes = [], []
        for factor in self.cropping_factors:
            crop, updated_box, _ = crop_using_bbox(image, bbox, factor)
            crops.append(crop)
            boxes.append(updated_box)

        crops = np.array(crops)
        boxes = np.array(boxes)
        print("[QueryEncoder.encode_image] crops shape:", crops.shape)
        print("[QueryEncoder.encode_image] boxes shape:", boxes.shape)

        tokens = generate_token_from_bbox(
            crops, boxes, self.ren, self.sam_ckpt, self.sam_cfg
        )

        if not torch.is_tensor(tokens):
            tokens = torch.tensor(tokens, dtype=torch.float32)
        else:
            tokens = tokens.to(torch.float32)
        tokens = F.normalize(tokens, dim=-1)
        print("[QueryEncoder.encode_image] tokens shape:", tokens.shape)
        return tokens


def encode_video_with_ren(ren, video_np, target_size=None):
    if not (isinstance(video_np, np.ndarray) and video_np.ndim == 4 and video_np.shape[-1] == 3):
        raise ValueError(
            f"Expected video_np as [T,H,W,3] uint8, got {type(video_np)} {getattr(video_np, 'shape', None)}"
        )
    if video_np.dtype != np.uint8:
        video_np = np.clip(video_np, 0, 255).astype(np.uint8)

    T, H, W, _ = video_np.shape
    print("[encode_video_with_ren] video shape:", video_np.shape)

    ren_out = ren(video_np, key_prefix='frame')

    obj_tokens = []
    obj_points = []
    fids = []

    keys = sorted(ren_out.keys(), key=lambda s: int(s.split('-')[-1]))
    print("[encode_video_with_ren] number of frames in ren_out:", len(keys))

    for k in keys:
        f = int(k.split('-')[-1])
        items = ren_out[k]
        for it in items:
            tok = it['region_feature']
            pt = it['point']
            tok_t = torch.from_numpy(tok) if isinstance(tok, np.ndarray) else torch.as_tensor(tok)
            if tok_t.ndim == 1:
                tok_t = tok_t.unsqueeze(0)
            obj_tokens.append(tok_t.squeeze(0).to(torch.float32))
            obj_points.append(torch.tensor(pt, dtype=torch.float32))
            fids.append(f)

    if not obj_tokens:
        print("[encode_video_with_ren] no object tokens found, returning empty tensors")
        return (
            torch.zeros((0, 1), dtype=torch.float32),
            torch.zeros((0, 2), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.int64),
            T,
        )

    obj_tokens = torch.stack(obj_tokens, dim=0)
    obj_points = torch.stack(obj_points, dim=0)
    frame_ids = torch.tensor(fids, dtype=torch.int64)

    print("[encode_video_with_ren] obj_tokens shape:", obj_tokens.shape)
    print("[encode_video_with_ren] obj_points shape:", obj_points.shape)
    print("[encode_video_with_ren] frame_ids shape:", frame_ids.shape)

    return obj_tokens, obj_points, frame_ids, T


def cosine_match_for_frames(query_tokens, obj_tokens, obj_points, frame_ids, num_frames, tau=5.0):
    """
    Compute cosine similarity between query_tokens [Q, D] and obj_tokens [N, D].
    For each frame f, pick the single best scoring object point.
    """

    if query_tokens.ndim == 1:
        query_tokens = query_tokens.unsqueeze(0)

    q = query_tokens.to(torch.float32)
    o = obj_tokens.to(torch.float32)

    print("[cosine_match_for_frames] raw q shape:", q.shape)
    print("[cosine_match_for_frames] raw o shape:", o.shape)
    print("[cosine_match_for_frames] obj_points shape:", obj_points.shape)
    print("[cosine_match_for_frames] frame_ids shape:", frame_ids.shape)
    print("[cosine_match_for_frames] num_frames:", num_frames)

    if q.shape[0] > 1:
        print("[cosine_match_for_frames] multiple query tokens detected. Q =", q.shape[0])

    # Only normalize object tokens here. Query tokens were normalized once in QueryEncoder
    o = F.normalize(o, dim=-1)

    sim = q @ o.T  # [Q, N]
    print("[cosine_match_for_frames] sim shape:", sim.shape)

    # Best query for each object
    best_obj_scores, best_q_idx = sim.max(dim=0)  # [N], [N]
    print("[cosine_match_for_frames] best_obj_scores shape:", best_obj_scores.shape)
    print("[cosine_match_for_frames] best_q_idx shape:", best_q_idx.shape)
    # debug
    def debug_topk_for_frame(frame_idx, topk=10):
        frame_mask = (frame_ids == frame_idx)
        idxs = frame_mask.nonzero(as_tuple=False).squeeze(1)

        if idxs.numel() == 0:
            print(f"no objects for frame {frame_idx}")
            return

        frame_scores = best_obj_scores[idxs]
        frame_pts = obj_points[idxs]

        top_scores, top_indices = torch.topk(frame_scores, k=min(topk, frame_scores.numel()))
        print(f"Top {topk} objects for frame {frame_idx}:")
        for rank in range(top_scores.numel()):
            s = float(top_scores[rank].item())
            i = idxs[top_indices[rank]].item()
            pt = frame_pts[top_indices[rank]]
            print(f"rank {rank}: score={s:.4f}, idx={i}, point={pt.tolist()}")
    debug_topk_for_frame(best_frame, topk=10)
    per_frame_points = []
    per_frame_best_scores = torch.full((num_frames,), -1e9, dtype=torch.float32)

    for f in range(num_frames):
        mask = (frame_ids == f)
        if not mask.any():
            per_frame_points.append({"frame": f, "x": 0, "y": 0})
            print(f"[cosine_match_for_frames] WARNING. no objects in frame {f}")
            continue

        idxs = mask.nonzero(as_tuple=False).squeeze(1)  # indices into obj_tokens
        frame_scores = best_obj_scores[idxs]
        frame_pts = obj_points[idxs].to(torch.float32)

        # Pick the single best scoring object in this frame. 
        best_idx_local = torch.argmax(frame_scores)
        best_score = frame_scores[best_idx_local]
        best_pt = frame_pts[best_idx_local]  # [y, x]

        y, x = float(best_pt[0].item()), float(best_pt[1].item())

        per_frame_best_scores[f] = float(best_score.item())
        per_frame_points.append({"frame": f, "x": int(round(x)), "y": int(round(y))})

    best_frame = int(torch.argmax(per_frame_best_scores).item())
    best_score = float(per_frame_best_scores[best_frame].item())

    print("[cosine_match_for_frames] per_frame_best_scores shape:", per_frame_best_scores.shape)
    print("[cosine_match_for_frames] best_frame:", best_frame, "best_score:", best_score)

    return best_frame, best_score, per_frame_points, per_frame_best_scores


def run_pipeline(config_path, video_path, image_path, out_path=None, ren_only=False, bbox=None):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    video = read_video_rgb(video_path)
    query = read_image_rgb(image_path)

    print("[run_pipeline] video shape:", video.shape)
    print("[run_pipeline] query shape:", query.shape)

    encoder = QueryEncoder(config)
    query_tokens = encoder.encode_image(query, bbox=bbox)
    print("[run_pipeline] query_tokens shape:", query_tokens.shape)

    ren = REN(config).eval()
    target_size = config['ren']['parameters']['image_resolution']
    obj_tokens, obj_points, frame_ids, T = encode_video_with_ren(ren, video, target_size)

    best_frame, best_score, frame_points, per_frame_best_scores = cosine_match_for_frames(
        query_tokens, obj_tokens, obj_points, frame_ids, T
    )

    frame_scores = [
        {"frame": int(f), "score": float(per_frame_best_scores[f].item())}
        for f in range(T)
    ]

    if ren_only:
        track_boxes = []
        H, W = video.shape[1:3]
        for d in frame_points:
            x = max(0, min(W - 1, int(d["x"])))
            y = max(0, min(H - 1, int(d["y"])))
            track_boxes.append({"frame": d["frame"], "x": x, "y": y, "w": 5, "h": 5})
        result = {
            "video": os.path.basename(video_path),
            "image": os.path.basename(image_path),
            "prompt_type": "point",
            "prompt_definition": "ren_best_point_per_frame",
            "best_frame": best_frame,
            "score": best_score,
            "frame_scores": frame_scores,
            "track": track_boxes,
        }
    else:
        H, W = video.shape[1:3]
        best_pt = next((p for p in frame_points if p["frame"] == best_frame), None)
        if best_pt is None:
            raise RuntimeError("Best frame point missing; cannot seed SAM2")

        seed_x = max(0, min(W - 1, int(best_pt["x"])))
        seed_y = max(0, min(H - 1, int(best_pt["y"])))
        half_w = 20
        half_h = 20
        x1 = max(0, seed_x - half_w)
        y1 = max(0, seed_y - half_h)
        x2 = min(W - 1, seed_x + half_w)
        y2 = min(H - 1, seed_y + half_h)
        seed_box = [x1, y1, x2, y2]
        seed_score = float(best_score)
        
        vqt = VisualQueryTracker(config)
        track_out = vqt.track(
            frames=video,
            selected_frame_ids=torch.tensor([best_frame]),
            selected_object_bboxes=torch.tensor([seed_box], dtype=torch.float32),
            selected_object_scores=torch.tensor([seed_score], dtype=torch.float32),
            query_timestep=torch.tensor(best_frame),
            get_tokens=False,
            top_p=float(config["visual_query"].get("tracking_top_p", 0.75)),
        )

        pred_track = []
        for step in track_out["predicted_track"]:
            pred_track.append({
                "frame": int(step["frame"]),
                "x": int(step["x"]),
                "y": int(step["y"]),
                "w": int(step["w"]),
                "h": int(step["h"]),
                "score": float(step["score"]) if "score" in step and step["score"] is not None else None,
            })

        per_frame_query_points = [
            {"frame": int(p["frame"]), "x": int(p["x"]), "y": int(p["y"])}
            for p in frame_points
        ]

        result = {
            "video": os.path.basename(video_path),
            "image": os.path.basename(image_path),
            "best_frame": best_frame,
            "score": best_score,
            "frame_scores": frame_scores,
            "track": pred_track,
            "query_points": per_frame_query_points,
        }

    out_path = out_path or os.path.join(os.path.dirname(video_path), "prediction.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Best frame: {best_frame}, Score: {best_score:.4f}")
    print(f"Output saved: {out_path}")


def parse_bbox_arg(bbox_str):
    """
    Parse a bbox string into [x1, y1, x2, y2].
    """
    if bbox_str is None:
        return None
    parts = bbox_str.replace(",", " ").split()
    if len(parts) != 4:
        raise ValueError(
            f"Invalid bbox format: {bbox_str}. expected 4 numbers like 'x1,y1,x2,y2'"
        )
    x1, y1, x2, y2 = map(int, parts)
    return [x1, y1, x2, y2]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--ren-only", action="store_true")
    parser.add_argument(
        "--bbox",
        type=str,
        default=None,
        help="Bounding box for the query image as 'x1,y1,x2,y2' in image coordinates",
    )
    args = parser.parse_args()

    try:
        bbox = parse_bbox_arg(args.bbox)
        run_pipeline(args.config, args.video, args.image, args.out, args.ren_only, bbox=bbox)
    except Exception as e:
        print(f"Error: {e}")
        raise
