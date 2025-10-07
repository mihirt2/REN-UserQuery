import os, cv2, json, yaml, argparse
import numpy as np
import torch
# SLIC: Selective Localization and Instance Calibration for Knowledge-Enhanced Car Damage Segmentation in Automotive Insurance
# Cluster pixels with similar 
from ren import REN 
from visual_query.models import CandidateSelector, CandidateRefiner, VisualQueryTracker
from visual_query.vq_utils import crop_using_bbox, generate_token_from_bbox, print_log

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def read_video_rgb(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    frames = []
    ok, frame = cap.read()
    while ok:
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ok, frame = cap.read()
    cap.release()
    return np.stack(frames, axis=0)  # [T,H,W,3], uint8

def read_image_rgb(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # [H,W,3], uint8

def build_batch(video_np: np.ndarray, query_img: np.ndarray, config: dict):
    """Build a single batch matching your pipeline’s expected keys/shapes."""
    T, H, W, _ = video_np.shape
    qH, qW = query_img.shape[:2]

    # Query tokens from the whole query image 
    ren = REN(config).eval()
    full_box = [0, 0, qW, qH]
    cfs = config['visual_query'].get('query_cropping_factors', [1.0])
    crop_imgs, crop_boxes = [], []
    for cf in cfs:
        cropped, upd_box, _ = crop_using_bbox(query_img, full_box, cf)
        crop_imgs.append(cropped)
        crop_boxes.append(upd_box)
    crop_imgs = np.array(crop_imgs)
    crop_boxes = np.array(crop_boxes)
    # Find a way to click on the points and have them be the queries
    query_tokens = generate_token_from_bbox(
        crop_imgs, crop_boxes, ren,
        config['data']['sam2_ckpt'],
        config['visual_query']['tracker_param'],
    )                                  # [Q,D] 
    query_tokens = query_tokens[None]  # [1,Q,D] (batch dimention, query objects, dimension)
    query_bboxes = torch.tensor([[full_box]], dtype=torch.int32)     
    query_frame_numbers = torch.tensor([[-1]], dtype=torch.int32)    # not in video
    query_timestep = torch.tensor([T - 1], dtype=torch.int32)

    # Video object tokens/points/frame_ids using REN 
    tokens_by_frame = ren(
        video_np,
        matching_tokens=query_tokens[0],
        matching_threshold=0.0,
        batch_size=config['visual_query'].get('encode_batch_size', 4),
        key_prefix='frame',
        cache_attn_maps=False,
    )

    obj_feats, obj_pts, fids = [], [], []
    for f in range(T):
        key = f'frame-{f}'
        if key not in tokens_by_frame:
            continue
        for item in tokens_by_frame[key]:
            obj_feats.append(item['region_feature'])
            obj_pts.append(item['point'])  # [y, x]
            fids.append([f])

    object_tokens = torch.tensor(np.array(obj_feats), dtype=torch.float32)[None]       # [1,N,D]
    object_points = torch.tensor(np.array(obj_pts),  dtype=torch.float32)[None]        # [1,N,2]
    frame_ids     = torch.tensor(np.array(fids),     dtype=torch.int32).flatten()[None]# [1,N]
    object_attn_mask = torch.ones((1, object_tokens.shape[1]), dtype=torch.int32)
    frames_tensor = torch.from_numpy(video_np)[None]                                    # [1,T,H,W,3]

    return {
        'object_tokens': object_tokens,
        'object_points': object_points,
        'object_attn_mask': object_attn_mask,
        'frame_ids': frame_ids,
        'frames': frames_tensor,
        'query_tokens': query_tokens,
        'query_bboxes': query_bboxes,
        'query_frame_numbers': query_frame_numbers,
        'query_timestep': query_timestep,
        'response_track': torch.zeros((1,0,5), dtype=torch.int32),  # no GT if uploaded
    }

def run_once(config, video_path, image_path):
    print(config)
    save_dir = os.path.join(config['visual_query']['save_dir'])
    os.makedirs(save_dir, exist_ok=True)
    print_log(f'Upload run: video="{video_path}", image="{image_path}"', save_dir)

    # Read inputs
    video_np = read_video_rgb(video_path)
    query_img = read_image_rgb(image_path)

    # Build batch
    batch = build_batch(video_np, query_img, config)

    # Modules
    selector = CandidateSelector()
    refiner  = CandidateRefiner(config)
    tracker  = VisualQueryTracker(config)

    # Selection
    sel = selector(
        batch['object_tokens'], batch['query_tokens'], batch['object_attn_mask'],
        batch['frame_ids'], batch['query_frame_numbers'][0][0],  # -1
        top_k=config['visual_query']['selection_top_k'],
        top_p=config['visual_query']['selection_top_p'],
        nms_threshold=config['visual_query']['nms_threshold'],
        nms_window=config['visual_query'].get('nms_window', 1),
    )

    sel_idxs   = sel['selected_object_idxs']
    sel_scores = sel['selected_object_scores']
    qmatch     = sel['query_match_idxs']

    sel_points = torch.gather(batch['object_points'], 1, sel_idxs.unsqueeze(2).expand(-1, -1, 2))
    sel_fids   = torch.gather(batch['frame_ids'],   1, sel_idxs)

    cand = {
        'frames': batch['frames'],
        'selected_object_points': sel_points,
        'selected_object_scores': sel_scores,
        'selected_object_idxs': sel_idxs,
        'selected_frame_ids': sel_fids,
        'query_tokens': batch['query_tokens'],
        'query_match_idxs': qmatch,
        'query_frame_numbers': batch['query_frame_numbers'],
        'query_bboxes': batch['query_bboxes'],
        'query_timestep': batch['query_timestep'],
        'response_track': batch['response_track'],
    }

    # Refinement 
    refined = refiner([cand], top_p=config['visual_query'].get('refinement_top_p', None))[0]

    # Tracking 
    trk = tracker.track(
        refined['frames'][0].numpy(),
        refined['selected_frame_ids'][0],
        refined['refined_object_bboxes'][0],
        refined['refined_object_scores'][0],
        refined['query_timestep'][0],
        get_tokens=False,
        top_p=config['visual_query']['tracking_top_p'],
    )

    out = {
        "video": os.path.basename(video_path),
        "image": os.path.basename(image_path),
        "score": float(trk['predicted_track_score']),
        "track": [{"frame": int(f), "x": int(x), "y": int(y), "w": int(w), "h": int(h)}
                  for (f,x,y,w,h) in trk['predicted_track']]
    }
    with open(os.path.join(save_dir, "prediction.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nScore: {out['score']:.4f} | Frames in track: {len(out['track'])}")
    print(f"Saved: {os.path.join(save_dir, 'prediction.json')}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--video", required=True, help="Path to video file")
    ap.add_argument("--image", required=True, help="Path to query image")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    run_once(config, args.video, args.image)

if __name__ == "__main__":
    main()
