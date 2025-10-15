import os, cv2, json, yaml, argparse
import numpy as np
import torch
import time
import torch.nn.functional as F
# SLIC: Selective Localization and Instance Calibration for Knowledge-Enhanced Car Damage Segmentation in Automotive Insurance
# Cluster pixels with similar 
from ren import REN 
from visual_query.models import CandidateSelector, CandidateRefiner, VisualQueryTracker
from visual_query.vq_utils import crop_using_bbox, generate_token_from_bbox, print_log

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import torch
import torch.nn.functional as F
import numpy as np

def encode_video_with_ren(ren, video_np, target_size):
    """
    Robustly encodes a video with REN and returns:
      object_tokens  [1, N, D]
      object_points  [1, N, 2]  (y,x in ORIGINAL H,W)
      frame_ids      [1, N]
    Handles REN.forward outputs of shape:
      - torch.Tensor [T,G,D] or [G,D]
      - list/tuple of [G,D] tensors (len == T)
      - dicts containing keys like 'aggregated_pred_tokens', 'aggregated_proj_tokens',
        'pred_tokens', 'proj_tokens' (possibly nested per-frame)
    """

    def _coerce_tokens_to_TGD(x):
        """Return torch.Tensor [T,G,D] from various REN outputs."""
        if torch.is_tensor(x):
            if x.ndim == 3:           # [T,G,D]
                return x
            if x.ndim == 2:           # [G,D]
                return x.unsqueeze(0) # -> [1,G,D]
            raise ValueError(f"Unexpected REN tensor shape {tuple(x.shape)}")

        if isinstance(x, dict):
            for k in ("aggregated_pred_tokens", "aggregated_proj_tokens",
                      "pred_tokens", "proj_tokens"):
                if k in x:
                    return _coerce_tokens_to_TGD(x[k])
            raise ValueError(f"REN dict output missing expected token keys: {x.keys()}")

        if isinstance(x, (list, tuple)):
            elems = []
            for e in x:
                if torch.is_tensor(e):
                    if e.ndim == 2:        # [G,D]
                        elems.append(e)
                    elif e.ndim == 3 and e.shape[0] == 1:  # [1,G,D]
                        elems.append(e[0])
                    else:
                        raise ValueError(f"Unexpected per-frame tensor shape {tuple(e.shape)}")
                elif isinstance(e, dict):
                    # same key search per frame
                    added = False
                    for k in ("aggregated_pred_tokens", "aggregated_proj_tokens",
                              "pred_tokens", "proj_tokens"):
                        if k in e:
                            v = e[k]
                            if torch.is_tensor(v) and v.ndim == 2:
                                elems.append(v)
                                added = True
                                break
                    if not added:
                        raise ValueError("Per-frame dict missing expected token keys.")
                else:
                    raise TypeError(f"Unexpected per-frame type {type(e)}")
            if len(elems) == 0:
                raise ValueError("Empty REN list/tuple output.")
            return torch.stack(elems, dim=0)  # [T,G,D]

        raise TypeError(f"Unexpected REN output type {type(x)}")

    assert isinstance(video_np, np.ndarray) and video_np.ndim == 4 and video_np.shape[-1] == 3
    T, H, W, _ = video_np.shape
    frames_t = torch.from_numpy(video_np).permute(0,3,1,2).float() / 255.0  # [T,3,H,W]

    device = next(ren.parameters()).device
    frames_t = frames_t.to(device, non_blocking=True)
    with torch.inference_mode():
        frames_rs = F.interpolate(frames_t, size=(target_size, target_size),
                                  mode="bilinear", align_corners=False)     # [T,3,S,S]
    with torch.inference_mode():
        ren_out = ren(frames_rs)    

    region_tokens = _coerce_tokens_to_TGD(ren_out)  # -> [T,G,D] (on device or cpu)
    if region_tokens.device != torch.device("cpu"):
        region_tokens = region_tokens.detach().to(torch.float32).cpu()

    T2, G, D = region_tokens.shape
    if T2 != T:
        # If REN ran on a subset or different ordering, we still proceed,
        # but frame_ids will reflect 0..T2-1
        T = T2

    # ---- build points in ORIGINAL H,W ----
    src_size = ren.config['ren']['parameters']['image_resolution']
    gp = ren.grid_points.to(torch.float32)  # [G,2] (y,x on src_size grid)
    scale_y = float(H) / float(src_size)
    scale_x = float(W) / float(src_size)
    base_pts = gp.clone()
    base_pts[:, 0] = base_pts[:, 0] * scale_y
    base_pts[:, 1] = base_pts[:, 1] * scale_x      # [G,2]

    pts = base_pts.unsqueeze(0).repeat(T, 1, 1)    # [T,G,2] float32
    fids = torch.arange(T, dtype=torch.int32).unsqueeze(1).repeat(1, G)  # [T,G]

    # ---- flatten ----
    object_tokens = region_tokens.reshape(1, T*G, D).contiguous()     # [1,N,D]
    object_points = pts.reshape(1, T*G, 2).contiguous()               # [1,N,2]
    frame_ids     = fids.reshape(1, T*G).contiguous()                 # [1,N]

    return object_tokens, object_points, frame_ids

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
    crop = crop_imgs[0]
    if isinstance(crop, torch.Tensor):
    # crop is torch: ensure shape and dtype
        if crop.ndim == 3 and crop.shape[-1] == 3:      # [H,W,3]
            crop_t = crop.permute(2, 0, 1).unsqueeze(0)
        elif crop.ndim == 3 and crop.shape[0] == 3:     # [3,H,W]
            crop_t = crop.unsqueeze(0)
        elif crop.ndim == 4:                             # already batched [B,3,H,W] or [B,H,W,3]
            if crop.shape[1] == 3:
                crop_t = crop
            elif crop.shape[-1] == 3:
                crop_t = crop.permute(0, 3, 1, 2)
            else:
                raise ValueError(f"Unexpected crop tensor shape {tuple(crop.shape)}")
        else:
            raise ValueError(f"Unexpected crop tensor ndim {crop.ndim}")
        crop_t = crop_t.float()
        if crop_t.max() > 1.0:  # likely uint8-typed originally
            crop_t = crop_t / 255.0
    elif isinstance(crop, np.ndarray):
            if crop.ndim != 3 or crop.shape[-1] != 3:
                raise ValueError(f"Expected numpy crop [H,W,3], got {crop.shape}")
            crop_t = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    else:
        raise TypeError(f"Unsupported crop type: {type(crop)}")
    target = int(config['ren']['parameters'].get('image_resolution', 308))
    crop_t = F.interpolate(crop_t, size=(target, target), mode="bilinear", align_corners=False)
    # bbox must be batched to match B=1
    crop_box = np.array([[0, 0, target, target]], dtype=np.int32)  # shape [1,4]
    # TODO: Implementing to click on the points and have them be the queries
    sam_ckpt = config['ren']['pretrained']['sam2_hiera_ckpt']
    sam_cfg  = config['visual_query'].get(
    'tracker_param',
    config['ren']['pretrained'].get('sam2_hiera_config')
)

    t0 = time.perf_counter()
    query_tokens = generate_token_from_bbox(
        crop_t, crop_box, ren,
        sam_ckpt,
        sam_cfg,
    )                                 
    t1 = time.perf_counter()
    print(f"[timing] SAM2 token generation: {t1 - t0:.2f}s")
    query_tokens = query_tokens[None]  # [1,Q,D] (batch dimention, query objects, dimensions)
    query_bboxes = torch.tensor([[full_box]], dtype=torch.int32)     
    query_frame_numbers = torch.tensor([[-1]], dtype=torch.int32)    # not in video
    query_timestep = torch.tensor([T - 1], dtype=torch.int32)
    t2 = time.perf_counter()
    target = int(config['ren']['parameters'].get('image_resolution', 308))
    object_tokens, object_points, frame_ids = encode_video_with_ren(ren, video_np, target)
    object_attn_mask = torch.ones((1, object_tokens.shape[1]), dtype=torch.int32)

    frames_tensor = torch.from_numpy(video_np)[None]  # [1,T,H,W,3]

    t3 = time.perf_counter()
    print(f"[timing] REN encode (DINOv2 + regions + matching): {t3 - t2:.2f}s")
    print(f"[timing] build_batch total: {t3 - t0:.2f}s")
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
        'response_track': torch.zeros((1,0,5), dtype=torch.int32),  
    }

def run_once(config, video_path, image_path):
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
