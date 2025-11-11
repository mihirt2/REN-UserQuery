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
from visual_query.models import CandidateSelector, CandidateRefiner, VisualQueryTracker
from visual_query.vq_utils import crop_using_bbox, generate_token_from_bbox, print_log

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
    return np.stack(frames).astype(np.uint8)

def read_image_rgb(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)

class QueryEncoder:
    def __init__(self, config):
        self.ren = REN(config)
        self.cropping_factors = config['visual_query']['query_cropping_factors']
        self.sam_ckpt = config['data']['sam2_ckpt']
        self.sam_cfg = config['visual_query']['tracker_param']

    def encode_image(self, image):
        h, w = image.shape[:2]
        bbox = [0, 0, w, h]
        crops, boxes = [], []
        for factor in self.cropping_factors:
            crop, updated_box, _ = crop_using_bbox(image, bbox, factor)
            crops.append(crop)
            boxes.append(updated_box)
        crops = np.array(crops)
        boxes = np.array(boxes)
        tokens = generate_token_from_bbox(crops, boxes, self.ren, self.sam_ckpt, self.sam_cfg)
        return torch.tensor(tokens, dtype=torch.float32) if not torch.is_tensor(tokens) else tokens


# REN Utilities
def encode_video_with_ren(ren, video_np, target_size):
    if video_np.dtype != np.uint8:
        video_np = np.clip(video_np, 0, 255).astype(np.uint8)
    ren_out = ren(video_np, key_prefix='frame')

    tokens, points, frame_ids = [], [], []
    for k in sorted(ren_out.keys(), key=lambda x: int(x.split('-')[-1])):
        frame_idx = int(k.split('-')[-1])
        for obj in ren_out[k]:
            tokens.append(torch.tensor(obj['region_feature'], dtype=torch.float32))
            points.append(torch.tensor(obj['point'], dtype=torch.float32))
            frame_ids.append(frame_idx)

    if not tokens:
        return (torch.zeros((0, 1)), torch.zeros((0, 2)), torch.zeros((0,), dtype=torch.long), 0)

    return (
        torch.stack(tokens),
        torch.stack(points),
        torch.tensor(frame_ids, dtype=torch.long),
        video_np.shape[0],
    )

def cosine_match_per_frame(query_tokens, obj_tokens, obj_points, frame_ids, num_frames):
    q = F.normalize(query_tokens.float(), dim=-1)
    o = F.normalize(obj_tokens.float(), dim=-1)
    sim = q @ o.T
    best_obj_scores, _ = sim.max(dim=0)

    per_frame_points = []
    per_frame_best_scores = torch.full((num_frames,), -1e9)
    for f in range(num_frames):
        mask = (frame_ids == f)
        if not mask.any():
            per_frame_points.append({"frame": f, "x": 0, "y": 0})
            continue
        idxs = mask.nonzero(as_tuple=False).squeeze(1)
        scores = best_obj_scores[idxs]
        best_idx = idxs[torch.argmax(scores)]
        y, x = obj_points[best_idx]
        per_frame_points.append({"frame": f, "x": int(round(float(x))), "y": int(round(float(y)))})
        per_frame_best_scores[f] = scores.max()

    best_frame = int(torch.argmax(per_frame_best_scores))
    best_score = float(per_frame_best_scores[best_frame])
    return best_frame, best_score, per_frame_points


def run_pipeline(config_path, video_path, image_path, out_path=None, ren_only=False):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    video = read_video_rgb(video_path)
    query = read_image_rgb(image_path)

    encoder = QueryEncoder(config)
    query_tokens = encoder.encode_image(query)

    ren = REN(config).eval()
    target_size = config['ren']['parameters']['image_resolution']
    obj_tokens, obj_points, frame_ids, T = encode_video_with_ren(ren, video, target_size)

    best_frame, best_score, frame_points = cosine_match_per_frame(
        query_tokens, obj_tokens, obj_points, frame_ids, T
    )

    if ren_only:
        # Save minimal REN-only output with (x,y) track points only
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
            "track": track_boxes
        }
    else:
        result = {
            "video": os.path.basename(video_path),
            "image": os.path.basename(image_path),
            "best_frame": best_frame,
            "score": best_score,
            "track": frame_points
        }

    out_path = out_path or os.path.join(os.path.dirname(video_path), "prediction.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Best frame: {best_frame}, Score: {best_score:.4f}")
    print(f"Output saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--ren-only", action="store_true", help="Run REN-only mode without full pipeline")
    args = parser.parse_args()

    try:
        run_pipeline(args.config, args.video, args.image, args.out, args.ren_only)
    except Exception as e:
        print(f"Error: {e}")
        raise
