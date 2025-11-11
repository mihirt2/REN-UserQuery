import os
import sys
import math
import cv2
import numpy as np
import itertools
from matplotlib import pyplot as plt
from matplotlib import patches
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from pycocotools import mask as mask_utils
import importlib.resources as ir
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import contextlib

sys.path.append('../segment_anything/')
from sam2.build_sam import build_sam2  # type: ignore
from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CenterPadding(torch.nn.Module):
    def __init__(self, multiple=14):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output


def extract_dino(model, images, batch_size=32, patch_length=8, layers=[11]):
    import torch
    import torchvision.transforms as T
    import torch.nn.functional as F

    assert len(layers) == 1, 'Implemented for single layer extraction only.'

    transform = T.Compose([
        T.ToTensor(),                      # float32 by default
        T.Resize((384, 512), antialias=True),
        lambda x: x.unsqueeze(0),
        CenterPadding(multiple=patch_length),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # --- keep model and inputs float32 on CPU ---
    model = model.to('cpu').float().eval()

    transformed_images = []
    for image in images:
        transformed_images.append(transform(image))    # float32
    transformed_images = torch.cat(transformed_images, dim=0)  # [N,3,H,W], float32 (cpu)

    features = []
    print("dbg DINO dtypes:", transformed_images.dtype)
    for i in range(0, transformed_images.shape[0], batch_size):
        # IMPORTANT: stay float32 on CPU; do NOT cast to bfloat16 here
        image_batch = transformed_images[i:(i + batch_size)].to(device='cpu', dtype=torch.float32)
        with torch.inference_mode():
            n = 12 - layers[0]
            features_out = model.get_intermediate_layers(image_batch, n=n)[0]
            features_out = features_out[:, 1:].cpu()
            B, _, C = features_out.size()
            H, W = image_batch.shape[2], image_batch.shape[3]
            patch_H, patch_W = math.ceil(H / patch_length), math.ceil(W / patch_length)
            features_out = features_out.permute(0, 2, 1).view(B, C, patch_H, patch_W)
            features.append(features_out)
    features = torch.cat(features, dim=0)
    return features.detach().cpu().to(torch.float32)



import torch.nn.functional as F  # make sure this import exists
# in vq_utils.py (module-level)
_BACKBONE = None
_BACKBONE_KIND = None
_BACKBONE_PATCH = None

def get_backbone(requested="dino_vitb8"):
    global _BACKBONE, _BACKBONE_KIND, _BACKBONE_PATCH
    if _BACKBONE is not None:
        return _BACKBONE, _BACKBONE_KIND, _BACKBONE_PATCH

    # CPU-aware choice
    if requested in ("dinov2_vitl14", "dinov2_vitb14"):
        requested = "dinov2_vits14"  # auto-downgrade on CPU

    if requested.startswith("dinov2"):
        model = torch.hub.load('facebookresearch/dinov2', requested).eval().to("cpu")
        kind, patch = "dinov2", 14
    else:
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8').eval().to("cpu")
        kind, patch = "dino", 8

    _BACKBONE, _BACKBONE_KIND, _BACKBONE_PATCH = model, kind, patch
    return _BACKBONE, _BACKBONE_KIND, _BACKBONE_PATCH
import importlib.resources as ir
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_dir
from sam2.build_sam import build_sam2_video_predictor

def _normalize_sam2_name(name_or_path: str) -> str:
    # Accept "sam2/sam2_hiera_l", "configs/sam2/sam2_hiera_l.yaml", or a full path
    name = name_or_path.replace(".yaml", "")
    if name.startswith("configs/"):
        name = name[len("configs/"):]
    if name.startswith("segment_anything/sam2/configs/"):
        name = name[len("segment_anything/sam2/configs/"):]
    return name  # e.g., "sam2/sam2_hiera_l"

def _build_sam2_video_predictor_safe(tracker_param: str, sam2_ckpt: str, device: str):
    """
    Hydra-safe builder for the SAM2 *video* predictor (has init_state / propagate APIs).
    """
    cfg_dir = str(ir.files("sam2").joinpath("configs"))
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        name = _normalize_sam2_name(tracker_param)
        return build_sam2_video_predictor(name, sam2_ckpt, device=device)

def extract_dino_v2(model, images, batch_size=128, patch_length=14, layers=[23]):
    def _as_numpy_hwc(img):
        if isinstance(img, np.ndarray):
            if img.ndim == 3 and img.shape[-1] == 3:
                arr = img
            elif img.ndim == 3 and img.shape[0] == 3:
                arr = np.transpose(img, (1, 2, 0))
            else:
                raise ValueError(f"extract_dino_v2: unsupported numpy shape {img.shape}")
            if arr.dtype != np.uint8:
                arr = arr.astype(np.float32)
                if arr.max() <= 1.0: arr = (arr * 255.0).clip(0, 255)
                else: arr = arr.clip(0, 255)
                arr = arr.astype(np.uint8)
            return arr
        if torch.is_tensor(img):
            return _to_numpy_rgb(img)
        raise TypeError(f"extract_dino_v2: unsupported image type {type(img)}")

    # Build a list of items
    if isinstance(images, np.ndarray) and images.ndim == 4:
        items = [images[i] for i in range(images.shape[0])]
    elif torch.is_tensor(images) and images.ndim == 4:
        items = [images[i] for i in range(images.shape[0])]
    else:
        items = list(images)

    transform = T.Compose([
        T.ToTensor(),                     # -> [3,H,W], float in [0,1]
        CenterPadding(multiple=patch_length),  # center-pad to multiples of 14
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    transformed_images = []
    for image in items:
        np_img = _as_numpy_hwc(image)
        tens = transform(np_img)           # [3,H',W'] where H',W' % 14 == 0 (intended)
        transformed_images.append(tens.unsqueeze(0))
    transformed_images = torch.cat(transformed_images, dim=0)  # [B,3,H',W']

    # EXTRA SAFEGUARD: enforce multiple-of-14 at batch level
    B, C, H, W = transformed_images.shape
    pad_h = (-H) % patch_length
    pad_w = (-W) % patch_length
    if pad_h or pad_w:
        # pad (left, right, top, bottom) = (0, pad_w, 0, pad_h)
        transformed_images = F.pad(transformed_images, (0, pad_w, 0, pad_h))

    features = []
    for i in range(0, transformed_images.shape[0], batch_size):
        image_batch = transformed_images[i:(i + batch_size)].to(device=device, dtype=torch.bfloat16)
        with torch.inference_mode():
            feats = model.get_intermediate_layers(image_batch, n=layers, reshape=True)
            features.append(torch.cat(feats, dim=1))
            torch.cuda.empty_cache()
    features = torch.cat(features, dim=0)
    return features.detach().to(torch.float32)


    # If images is a batch array/tensor, iterate over first dimension
    items = []
    if isinstance(images, np.ndarray) and images.ndim == 4:
        items = [images[i] for i in range(images.shape[0])]
    elif torch.is_tensor(images) and images.ndim == 4:
        items = [images[i] for i in range(images.shape[0])]
    else:
        # assume iterable of images
        items = list(images)

    transform = T.Compose([
        T.ToTensor(),
        # keep original spatial size; padding handles patch grid
        CenterPadding(multiple=patch_length),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    transformed_images = []
    for image in items:
        np_img = _as_numpy_hwc(image)           # HWC uint8
        tens = transform(np_img)                # [3,H,W], float
        transformed_images.append(tens.unsqueeze(0))
    transformed_images = torch.cat(transformed_images, dim=0)  # [B,3,H,W]

    features = []
    for i in range(0, transformed_images.shape[0], batch_size):
        image_batch = transformed_images[i:(i + batch_size)].to(device=device, dtype=torch.bfloat16)
        with torch.inference_mode():
            # DINOv2 returns list of features when reshape=True
            feats = model.get_intermediate_layers(image_batch, n=layers, reshape=True)
            features.append(torch.cat(feats, dim=1))
            torch.cuda.empty_cache()
    features = torch.cat(features, dim=0)
    return features.detach().to(torch.float32)



def extract_sam(model, images, batch_size=4):
    transform = T.Compose([T.ToTensor(),
                           T.Resize((1024, 1024), antialias=True),
                           lambda x: x.unsqueeze(0),
                           T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    
    transformed_images = []
    for image in images:
        transformed_images.append(transform(image))
    transformed_images = torch.cat(transformed_images, dim=0)
    
    features = []
    for i in range(0, transformed_images.shape[0], batch_size):
        image_batch = transformed_images[i:(i + batch_size)].to(device=device, dtype=torch.bfloat16)
        with torch.inference_mode():
            features_out = model(image_batch).cpu()
            features.append(features_out)
            torch.cuda.empty_cache()
    features = torch.cat(features, dim=0)
    return features.detach().cpu().to(torch.float32)


def extract_sam2(model, images, batch_size=16):
    transform = T.Compose([T.ToTensor(),
                           T.Resize((1024, 1024), antialias=True),
                           lambda x: x.unsqueeze(0),
                           T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    
    transformed_images = []
    for image in images:
        transformed_images.append(transform(image))
    transformed_images = torch.cat(transformed_images, dim=0)
    
    features = []
    for i in range(0, transformed_images.shape[0], batch_size):
        image_batch = transformed_images[i:(i + batch_size)].to(device=device, dtype=torch.bfloat16)
        with torch.inference_mode():
            features_out = model(image_batch)['vision_features'].cpu()
            features.append(features_out)
            torch.cuda.empty_cache()
    features = torch.cat(features, dim=0)
    return features.detach().cpu().to(torch.float32)

import os
from hydra import initialize, initialize_config_dir, compose
from omegaconf import OmegaConf

# in visual_query/vq_utils.py (where you defined resolve_sam2_cfg)
from hydra import initialize, initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import os, importlib.resources as ir

import contextlib

def _to_numpy_rgb(frame):
    """Accepts torch tensor [H,W,3] or [3,H,W] or [B,3,H,W] (B must be 1), or np.ndarray.
       Returns np.ndarray [H,W,3] uint8 in RGB."""
    if isinstance(frame, np.ndarray):
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"Expected numpy [H,W,3], got {frame.shape}")
        if frame.dtype != np.uint8:
            # assume 0..1 or 0..255 floats/ints; clamp and cast
            arr = np.clip(frame, 0, 255).astype(np.float32)
            if arr.max() <= 1.0:
                arr = arr * 255.0
            return arr.astype(np.uint8)
        return frame

    if isinstance(frame, torch.Tensor):
        t = frame.detach().cpu()
        # handle batch
        if t.ndim == 4:
            if t.shape[0] != 1:
                raise ValueError(f"Batch >1 not supported here, got {tuple(t.shape)}")
            t = t[0]
        # CHW or HWC
        if t.ndim == 3 and t.shape[0] == 3:     # [3,H,W] -> [H,W,3]
            t = t.permute(1, 2, 0)
        elif t.ndim == 3 and t.shape[-1] == 3:  # already [H,W,3]
            pass
        else:
            raise ValueError(f"Unexpected tensor shape {tuple(t.shape)}")
        arr = t.numpy()
        # scale to uint8 if needed
        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            mx = float(arr.max()) if arr.size else 1.0
            # if already in 0..1 range, scale up; otherwise clamp to 0..255
            if mx <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    raise TypeError(f"Unsupported frame type: {type(frame)}")


def _build_sam2_predictor(tracker_param: str, sam2_ckpt: str, device: str):
    """
    tracker_param: config name relative to installed sam2/configs, e.g. 'sam2.1/sam2.1_hiera_l'
                   (NO 'configs/' prefix, NO '.yaml' suffix)
    """
    name = tracker_param.replace(".yaml", "")
    if name.startswith("configs/"):  # avoid sam2/configs/configs/...
        name = name[len("configs/"):]

    cfg_dir = str(ir.files("sam2").joinpath("configs"))  # .../site-packages/sam2/configs

    # (re)initialize Hydra so compose() inside build_sam2 works *now*
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        sam2_model = build_sam2(name, sam2_ckpt, device=device)
    return SAM2ImagePredictor(sam2_model)

def resolve_sam2_cfg(tracker_param: str):
    """
    Accepts either a config NAME like 'sam2.1/sam2.1_hiera_l' or
    an absolute FILE PATH to a .yaml. Returns an OmegaConf.
    """
    # 1) If it's a real file path, load directly (no Hydra compose)
    if os.path.isfile(tracker_param):
        return OmegaConf.load(tracker_param)

    # 2) Otherwise compose relative to installed sam2/configs directory
    cfg_dir = str(ir.files("sam2").joinpath("configs"))
    name = tracker_param.replace(".yaml", "")
    if name.startswith("configs/"):
        name = name[len("configs/"):]

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        return compose(config_name=name)
def extract_image_features(images, config):
    feature_extractor = config['visual_query'].get('feature_extractor') \
                        or config['ren']['parameters']['feature_extractor']

    if feature_extractor in ('dino', 'dino_vitb8'):
        print('Extracting features using DINO')
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8').eval().to('cpu').float()
        # patch = 8 for vitb8
        return extract_dino(model, images, batch_size=1, patch_length=8, layers=[11])

    elif feature_extractor in ('dinov2', 'dinov2_vits14', 'dinov2_vitl14'):
        print('Extracting features using DINOv2')
        name = 'dinov2_vits14' if 'vits14' in feature_extractor or feature_extractor == 'dinov2' else 'dinov2_vitl14'
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8').eval().to('cpu').float()
        patch_len = 14
        return extract_dino_v2(model, images, batch_size=1, patch_length=patch_len, layers=[23])


def upsample_feature(frame_features, new_h, new_w, padded_h, padded_w, upsampling_method='bilinear'):
    if upsampling_method == 'bilinear':
        upsampled_feature = torch.nn.functional.interpolate(frame_features, size=[padded_h, padded_w], mode='bilinear')
        upsampled_feature = T.CenterCrop((new_h, new_w)) (upsampled_feature).squeeze(dim=0)
    else:
        raise ValueError(f'{upsampling_method} is not a valid upsampling method.')
    return upsampled_feature


def refine_mask(mask):
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    mask = mask.astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    if component_areas.size == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    largest_component_label = 1 + np.argmax(component_areas)
    refined_mask = np.zeros_like(mask)
    refined_mask[labels == largest_component_label] = 1
    return refined_mask

def _grid_from_lengths(n_tokens: int, mask_hw, patch_size: int):
    """
    Infer (G_h, G_w) from n_tokens and mask size/patch size.
    For square-ish grids we try round numbers around H/patch_size and W/patch_size.
    """
    import math
    H, W = mask_hw
    # expected grids near these
    gh_guess = max(1, round(H / patch_size))
    gw_guess = max(1, round(W / patch_size))

    # try a few nearby factors to match n_tokens
    candidates = []
    for dh in (-1, 0, 1, 2):
        for dw in (-1, 0, 1, 2):
            gh = max(1, gh_guess + dh)
            gw = max(1, gw_guess + dw)
            if gh * gw == n_tokens:
                candidates.append((gh, gw))
    if candidates:
        # prefer the one closest to ratio H/W
        target_ratio = H / max(1, W)
        gh, gw = min(candidates, key=lambda p: abs((p[0] / max(1, p[1])) - target_ratio))
        return gh, gw

    # fallback: square-ish
    r = int(round(n_tokens ** 0.5))
    if r * r == n_tokens:
        return r, r

    # last resort: 1 x n
    return 1, n_tokens


def _unpack_obj(o):
    """
    Returns (feat[1,D], point or None, attn_or_None) for:
      - dict with keys 'region_feature', 'point' (optional 'attn_map')
      - (feature, point) tuple/list
      - raw feature tensor
    """
    import torch
    if isinstance(o, dict):
        feat = o.get('region_feature', o.get('feature', None))
        pt   = o.get('point', None)
        attn = o.get('attn_map', None)
        if feat is None:
            return None, None, None
        return torch.as_tensor(feat).unsqueeze(0), pt, attn
    if isinstance(o, (list, tuple)) and len(o) >= 2:
        feat, pt = o[0], o[1]
        return torch.as_tensor(feat).unsqueeze(0), pt, None
    if torch.is_tensor(o):
        # raw [D] or [1,D]
        return (o.unsqueeze(0) if o.ndim == 1 else o), None, None
    return None, None, None


def _normalize_point(p):
    """Return (py, px) ints or None."""
    if p is None:
        return None
    import numpy as np, torch
    if isinstance(p, torch.Tensor):
        p = p.detach().cpu().numpy()
    p = np.asarray(p).squeeze()
    if p.ndim == 0 or p.shape[0] < 2:
        return None
    return int(p[0]), int(p[1])


def get_sam_regions(sam, frames, bboxes=None, input_points=None, img_resolution=1024, batch_size=4,
                    dense_point_grid_size=20, process_points_using_bbox=False, preprocess_input_points=False,
                    multimask_output=True):
    def preprocess_image(image, height, width):
        trans = T.Compose([T.Resize((height, width), antialias=True)])
        image = torch.as_tensor(image).to(device)
        return trans(image.permute(2, 0, 1))

    def preprocess_bbox(bbox, old_size, new_size):
        x, y, width, height = bbox
        old_h, old_w = old_size
        new_h, new_w = new_size
        scale_w = new_w / old_w
        scale_h = new_h / old_h
        new_x_left = x * scale_w
        new_y_top = y * scale_h
        new_width = width * scale_w
        new_height = height * scale_h
        return torch.tensor([new_x_left, new_y_top, new_width, new_height])

    def preprocess_points(points, old_size, new_size):
        new_points = []
        for point in points:
            y, x = point
            old_h, old_w = old_size
            new_h, new_w = new_size
            scale_w = new_w / old_w
            scale_h = new_h / old_h
            new_x = x * scale_w
            new_y = y * scale_h
            new_points.append(torch.as_tensor([new_x.item(), new_y.item()], dtype=torch.int64))
        return torch.stack(new_points, dim=0).to(device)

    def postprocess_mask(mask, height, width):
        trans = T.Compose([T.Resize((height, width), antialias=True)])
        return trans(mask[None])[0]
    
    masks = {}
    with torch.inference_mode():
        for i in range(0, len(frames), batch_size):
            frames_batch = frames[i:i + batch_size]
            bboxes_batch = bboxes[i:i + batch_size]

            batched_input = []
            for frame, bbox in zip(frames_batch, bboxes_batch):
                processed_frame = preprocess_image(frame, img_resolution, img_resolution)
                processed_bbox = preprocess_bbox(bbox, (frame.shape[0], frame.shape[1]), (img_resolution, img_resolution))
                if preprocess_input_points:
                    input_points = preprocess_points(input_points, (frame.shape[0], frame.shape[1]), (img_resolution, img_resolution))
                if process_points_using_bbox:
                    x, y, width, height = processed_bbox
                    margin = 2 * img_resolution // dense_point_grid_size
                    processed_input_points = input_points[
                        (input_points[:, 0] >= x - margin) & (input_points[:, 0] <= x + width + margin) &
                        (input_points[:, 1] >= y - margin) & (input_points[:, 1] <= y + height + margin)
                    ]
                else:
                    processed_input_points = input_points
                processed_input_labels = torch.tensor([1 for _ in range(processed_input_points.shape[0])]).to(device)

                batched_input.append({
                    'image': processed_frame,
                    'point_coords': processed_input_points,
                    'point_labels': processed_input_labels,
                    'original_size': frame.shape[:-1]
                })

            segmentations = sam.individual_forward(batched_input, multimask_output=multimask_output)
            for j, frame_masks in enumerate(segmentations):
                masks[f'frame-{i + j}'] = []
                for mask in frame_masks:
                    postprocessed_mask = postprocess_mask(mask.cpu(), frames[j].shape[0], frames[j].shape[1])
                    refined_mask = refine_mask(postprocessed_mask)
                    masks[f'frame-{i + j}'].append({
                        'segmentation': refined_mask,
                    })
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
    return masks


def get_sam_region_from_bbox(sam2_ckpt, tracker_param, frames, bboxes):
    masks = []
    predictor = _build_sam2_predictor(tracker_param, sam2_ckpt, device)

    USE_CUDA = torch.cuda.is_available()
    AMP_CTX = (torch.autocast("cuda", dtype=torch.bfloat16) if USE_CUDA else contextlib.nullcontext())

    for frame, bbox in zip(frames, bboxes):
        img = _to_numpy_rgb(frame)          # <-- convert here
        # optional: slight blur; cv2 expects BGR
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_bgr = cv2.GaussianBlur(img_bgr, (3, 3), 0)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        with torch.inference_mode(), AMP_CTX:
            predictor.set_image(img)
            x, y, w, h = map(int, bbox)     # make sure ints
            box = np.array([x, y, x + w, y + h])[None]
            mask, _, _ = predictor.predict(
                point_coords=None, point_labels=None, box=box, multimask_output=False
            )
        masks.append(refine_mask(mask[0]))
    return masks


def get_sam_region_from_points(sam2_ckpt, tracker_param, frames, points):
    masks = []
    predictor = _build_sam2_predictor(tracker_param, sam2_ckpt, device)

    USE_CUDA = torch.cuda.is_available()
    AMP_CTX = (torch.autocast("cuda", dtype=torch.bfloat16) if USE_CUDA else contextlib.nullcontext())

    for frame, point in zip(frames, points):
        img = _to_numpy_rgb(frame)  # <-- convert here
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_bgr = cv2.GaussianBlur(img_bgr, (3, 3), 0)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        y, x = map(int, point)
        with torch.inference_mode(), AMP_CTX:
            predictor.set_image(img)
            # build a local box around the clicked point (limit SAM2 region)
# size ~ 30% of min(H,W); tune between 0.2–0.4
        box_half = int(0.3 * min(img.shape[0], img.shape[1]) * 0.5)
        x1 = max(0, x - box_half); y1 = max(0, y - box_half)
        x2 = min(img.shape[1]-1, x + box_half); y2 = min(img.shape[0]-1, y + box_half)

        mask, _, _ = predictor.predict(
            point_coords=np.array([[x, y]]),
            point_labels=np.array([1]),
            box=np.array([[x1, y1, x2, y2]]),   # <-- constrain
            multimask_output=True
        )
        masks.append([refine_mask(m) for m in mask])
    return masks


def generate_token_from_bbox(frames, bboxes, region_encoder, sam2_ckpt, tracker_param, return_attn_maps=False):
    """
    Build one token per input frame using SAM2 masks + region_encoder output.
    Robust to obj_list being a Tensor/list/tuple/None. Falls back to pooled tokens
    if no selection is possible across all frames.

    Requires helpers: _unpack_obj, _normalize_point, _grid_from_lengths,
    _to_numpy_rgb, get_sam_region_from_bbox, get_sam_pooled_tokens.
    """
    import torch, numpy as np, cv2

    # ---- safe-iteration helpers ----
    def _iter_objs(obj_list):
        if obj_list is None:
            return []
        if isinstance(obj_list, (list, tuple)):
            return obj_list
        if torch.is_tensor(obj_list):
            if obj_list.ndim == 2:   # [N, D]
                return [obj_list[i] for i in range(obj_list.shape[0])]
            if obj_list.ndim == 1:   # [D]
                return [obj_list]
            return []
        return []

    def _len_objs(obj_list):
        if obj_list is None:
            return 0
        if isinstance(obj_list, (list, tuple)):
            return len(obj_list)
        if torch.is_tensor(obj_list):
            if obj_list.ndim == 2:
                return obj_list.shape[0]
            if obj_list.ndim == 1:
                return 1
            return 0
        return 0

    # 1) Encode frames via region_encoder (REN)
    enc = region_encoder(frames)

    # 2) SAM2 masks from the provided bboxes
    masks = get_sam_region_from_bbox(sam2_ckpt, tracker_param, frames, bboxes)

    # 3) Normalize encoder output into a per-frame list
    if isinstance(enc, dict):
        items = sorted(enc.items(), key=lambda kv: int(str(kv[0]).split('-')[-1]))
        per_frame_obj_lists = [v for _, v in items]

    elif isinstance(enc, (list, tuple)):
        # already per-frame lists
        per_frame_obj_lists = list(enc)

    elif torch.is_tensor(enc):
        # Accept [T,G,D] or [G,D]
        if enc.ndim == 3:
            # build per-frame object lists of raw feature tensors [D]
            T, G, D = enc.shape
            per_frame_obj_lists = [[enc[t, g] for g in range(G)] for t in range(T)]
        elif enc.ndim == 2:
            # single frame [G,D]
            G, D = enc.shape
            per_frame_obj_lists = [[enc[g] for g in range(G)]]
        else:
            raise ValueError(f"Unexpected encoder tensor shape {tuple(enc.shape)}")

    else:
        raise TypeError(f"Unexpected encoder output type: {type(enc)}")

    n = min(len(per_frame_obj_lists), len(masks))
    if n == 0:
        raise RuntimeError("No frames/masks to process in generate_token_from_bbox")

    tokens, attn_maps = [], []
    any_selected = False

    # 4) Per-frame selection
    for obj_list, mask in zip(per_frame_obj_lists[:n], masks[:n]):
        m = np.asarray(mask, dtype=np.uint8)
        if m.size == 0 or m.sum() == 0:
            tokens.append(None)
            if return_attn_maps: attn_maps.append(None)
            continue

        # distance transform for this frame's mask
        dt = cv2.distanceTransform((m > 0).astype(np.uint8), cv2.DIST_L2, 5)

        best, best_attn, best_d = None, None, -1.0

        # Pass 1: objects that carry explicit points
        has_any_point = False
        for o in _iter_objs(obj_list):
            feat, pt, attn = _unpack_obj(o)
            if feat is None or pt is None:
                continue
            has_any_point = True
            norm = _normalize_point(pt)
            if norm is None:
                continue
            py, px = norm
            if not (0 <= py < m.shape[0] and 0 <= px < m.shape[1]):
                continue
            if not m[py, px]:
                continue
            d = float(dt[py, px])
            if d > best_d:
                best, best_attn, best_d = torch.as_tensor(feat), attn, d

        # Pass 2: raw feature tensors without points -> place on inferred grid
        if best is None and (not has_any_point):
            n_feat = _len_objs(obj_list)
            if n_feat > 0:
                gh, gw = _grid_from_lengths(n_feat, m.shape, 14)
                ys = np.linspace(0.5 * (m.shape[0] / gh), m.shape[0] - 0.5 * (m.shape[0] / gh), gh).astype(int)
                xs = np.linspace(0.5 * (m.shape[1] / gw), m.shape[1] - 0.5 * (m.shape[1] / gw), gw).astype(int)
                for idx, o in enumerate(_iter_objs(obj_list)):
                    feat, _, attn = _unpack_obj(o)
                    if feat is None:
                        continue
                    gy, gx = divmod(idx, gw)
                    py, px = ys[min(gy, gh - 1)], xs[min(gx, gw - 1)]
                    if not m[py, px]:
                        continue
                    d = float(dt[py, px])
                    if d > best_d:
                        best, best_attn, best_d = torch.as_tensor(feat), attn, d

        tokens.append(best)  # may be None for this frame
        if return_attn_maps:
            attn_maps.append(best_attn)
        any_selected = any_selected or (best is not None)

        # 5) Fallback: if nothing selected across all frames, pool features inside SAM masks
    if not any_selected:
        # --- normalize frames to a batched shape ---
        if torch.is_tensor(frames):
            if frames.ndim == 3:                     # [3,H,W] or [H,W,3]
                frames_batched = frames.unsqueeze(0)
            elif frames.ndim == 4:                   # [N,3,H,W] or [N,H,W,3]
                frames_batched = frames
            else:
                raise ValueError(f"Unsupported tensor frames shape {tuple(frames.shape)}")
        elif isinstance(frames, np.ndarray):
            if frames.ndim == 3:                     # [H,W,3]
                frames_batched = frames[None, ...]
            elif frames.ndim == 4:                   # [N,H,W,3]
                frames_batched = frames
            else:
                raise ValueError(f"Unsupported numpy frames shape {frames.shape}")
        else:
            raise TypeError(f"Unsupported frames type {type(frames)}")

        # --- convert every frame to numpy HWC uint8 so pooled path sees [N,H,W,3] ---
        frames_np_list = []
        for i in range(frames_batched.shape[0]):
            frames_np_list.append(_to_numpy_rgb(frames_batched[i]))
        frames_np = np.stack(frames_np_list, axis=0)   # [N,H,W,3]

        # --- normalize bboxes to [N,4] to match frames_np ---
        N = frames_np.shape[0]
        if torch.is_tensor(bboxes):
            bb = bboxes.detach().cpu().numpy()
        elif isinstance(bboxes, np.ndarray):
            bb = bboxes
        else:
            bb = np.asarray(bboxes)

        if bb.ndim == 1 and bb.shape[0] == 4:
            bboxes_batched = np.repeat(bb[None, :], N, axis=0)
        elif bb.ndim == 2 and bb.shape[0] in (1, N):
            bboxes_batched = bb if bb.shape[0] == N else np.repeat(bb, N, axis=0)
        else:
            raise ValueError(f"Unsupported bboxes shape for fallback: {bb.shape}")

        # Choose a default feature extractor for fallback (matches your helper)
        fallback_cfg = {"visual_query": {"feature_extractor": "dinov2"}}
        patch_size = 14

        pooled = get_sam_pooled_tokens(
            frames=frames_np,                 # <-- guaranteed [N,H,W,3] numpy
            bboxes=bboxes_batched,            # [N,4]
            sam2_ckpt=sam2_ckpt,
            tracker_param=tracker_param,
            patch_size=patch_size,
            config=fallback_cfg,
        )  # -> [N, D]
        if return_attn_maps:
            return pooled, [None] * pooled.shape[0]
        return pooled


    # 6) Replace None with zeros and concatenate
    ref = next((t for t in tokens if t is not None), None)
    if ref is None:
        raise RuntimeError("No token could be selected for any frame (post-fallback).")
    zero_like = torch.zeros_like(ref)
    tokens = [t if t is not None else zero_like for t in tokens]
    tokens = torch.cat(tokens, dim=0)

    if return_attn_maps:
        return tokens, attn_maps
    return tokens




def sliding_window_cropping(image, crop_size, overlap=0.2):
    image_height, image_width, _ = image.shape
    crop_height, crop_width = crop_size

    stride_x = int(crop_width * (1 - overlap))
    stride_y = int(crop_height * (1 - overlap))

    crops, crop_starts = [], []
    for y in range(0, image_height - crop_height + 1, stride_y):
        for x in range(0, image_width - crop_width + 1, stride_x):
            crop = image[y:y + crop_height, x:x + crop_width]
            crop = cv2.resize(crop, (image_width, image_height), interpolation=cv2.INTER_LANCZOS4)
            crops.append(crop)
            crop_starts.append([y, x])
    return crops, crop_starts


def mask_to_bbox(mask):
    """
    Input: mask as bool/0-1 array HxW (or tensor).
    Output: [x, y, w, h] with w,h >= 1 (ints).
    """
    import numpy as np, torch
    if torch.is_tensor(mask):
        m = mask.detach().cpu().numpy()
    else:
        m = np.asarray(mask)
    m = (m > 0).astype(np.uint8)

    ys, xs = np.where(m)
    if ys.size == 0:
        return [0, 0, 1, 1]  # fallback tiny box

    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())

    # +1 so a single pixel still yields size 1
    w = max(1, int(round(x2 - x1 + 1)))
    h = max(1, int(round(y2 - y1 + 1)))
    return [x1, y1, w, h]




def get_sam_pooled_tokens(frames, bboxes, sam2_ckpt, tracker_param, patch_size, config, chunk_size=8):
    all_tokens = []
    for i in range(0, frames.shape[0], chunk_size):
        frames_chunk = frames[i:i + chunk_size]
        bboxes_chunk = bboxes[i:i + chunk_size]

        # Generate the masks
        masks = get_sam_region_from_bbox(sam2_ckpt, tracker_param, frames_chunk, bboxes_chunk)

        # Generate the features
        frames_features = extract_image_features(frames_chunk, config)
        new_h, new_w = frames_chunk.shape[1], frames_chunk.shape[2]
        padded_h = math.ceil(new_h / patch_size) * patch_size
        padded_w = math.ceil(new_w / patch_size) * patch_size
        frames_features = upsample_feature(frames_features.cpu(), new_h, new_w, padded_h, padded_w)
        if len(frames_features.shape) == 3:
            frames_features = frames_features[None]

        # Generate query tokens
        for mask, frame_features in zip(masks, frames_features):
            r_1, r_2 = np.where(mask == 1)
            features = frame_features[:, r_1, r_2]
            feature_dims = features.shape[0]
            token = features.reshape(feature_dims, -1).mean(1)[None]
            all_tokens.append(token)
        torch.cuda.empty_cache()
    tokens = torch.cat(all_tokens)
    return tokens


def point_to_bbox(frame, object_point, query_tokens, query_frames, query_bboxes, sam2_ckpt, tracker_param, patch_size, config):
    if query_tokens is None:
        assert (query_frames is not None) and (query_bboxes is not None)
        query_tokens = get_sam_pooled_tokens(query_frames, query_bboxes, sam2_ckpt, tracker_param, patch_size, config)

    # Generate candidate mask
    points = np.array([object_point], dtype=np.float32)
    object_masks = get_sam_region_from_points(sam2_ckpt, tracker_param, frame[None], points)[0]
    
    # Generate candidate features
    frame_features = extract_image_features(frame[None], config)
    new_h, new_w = frame.shape[0], frame.shape[1]
    padded_h = math.ceil(new_h / patch_size) * patch_size
    padded_w = math.ceil(new_w / patch_size) * patch_size
    frame_features = upsample_feature(frame_features.cpu(), new_h, new_w, padded_h, padded_w)

    # Get candidate tokens
    candidate_similarities = []
    for object_mask in object_masks:
        r_1, r_2 = np.where(object_mask == 1)
        candidate_features = frame_features[:, r_1, r_2]
        feature_dims = candidate_features.shape[0]
        candidate_token = candidate_features.reshape(feature_dims, -1).mean(1)[None]
        x = F.normalize(candidate_token, p=2, dim=1)
        y = F.normalize(query_tokens, p=2, dim=1)
        cosine_scores = torch.mm(x, y.T)
        candidate_similarity = torch.max(cosine_scores, dim=1)[0].item()
        candidate_similarities.append(candidate_similarity)

    # Get object bbox
    object_mask = object_masks[candidate_similarities.index(max(candidate_similarities))]
    object_bbox = mask_to_bbox(object_mask)
    torch.cuda.empty_cache()
    return object_bbox


def get_cropping_factor(object_bbox, frame_size, cropping_margin_expansion, max_cropping_factor=2.0):
    object_width, object_height = object_bbox[2], object_bbox[3]
    frame_width, frame_height = frame_size
    width_cropping_factor = frame_width / (object_width * cropping_margin_expansion)
    height_cropping_factor = frame_height / (object_height * cropping_margin_expansion)
    cropping_factor = min(width_cropping_factor, height_cropping_factor)
    cropping_factor = np.clip(cropping_factor, 1.0, max_cropping_factor)
    return cropping_factor


def iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    intersect_x1 = max(x1, x2)
    intersect_y1 = max(y1, y2)
    intersect_x2 = min(x1 + w1, x2 + w2)
    intersect_y2 = min(y1 + h1, y2 + h2)
    intersection_area = max(0, intersect_x2 - intersect_x1) * max(0, intersect_y2 - intersect_y1)

    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - intersection_area

    iou = intersection_area / union_area
    return iou


def visualize_image_and_bbox(image, bbox, save_path):
    plt.imshow(image)
    plt.axis('off')
    x, y, width, height = bbox
    rectangle = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rectangle)
    plt.savefig(save_path)
    plt.clf()


def crop_using_bbox(image, bbox, cropping_factor=1.5):
    if not torch.is_tensor(image):
        image = torch.tensor(image)

    # Get bounding box information
    bbox_x, bbox_y, bbox_w, bbox_h = bbox
    bbox_center_x = bbox_x + bbox_w // 2
    bbox_center_y = bbox_y + bbox_h // 2
    
    # Get width and height of the crop
    img_height, img_width = image.size()[:2]
    cropping_factor = min(cropping_factor, img_width / bbox_w, img_height / bbox_h)
    new_width = img_width // cropping_factor
    new_height = img_height // cropping_factor

    # Get the crop with bounding box at its center
    crop_left = int(max(0, bbox_center_x - new_width // 2))
    crop_left = int(min(crop_left, img_width - new_width))
    crop_top = int(max(0, bbox_center_y - new_height // 2))
    crop_top = int(min(crop_top, img_height - new_height))
    crop_right = int(crop_left + new_width)
    crop_bottom = int(crop_top + new_height)
    cropped_image = image[crop_top:crop_bottom, crop_left:crop_right, :].numpy()
    cropped_image = cv2.resize(cropped_image, (img_width, img_height), interpolation=cv2.INTER_LANCZOS4)
    cropped_image = torch.tensor(cropped_image)

    # Get the updated bounding box
    scale_x = img_width / new_width
    scale_y = img_height / new_height
    updated_bbox_x = max(0, bbox_x - crop_left) * scale_x
    updated_bbox_y = max(0, bbox_y - crop_top) * scale_y
    updated_bbox_w = min(bbox_w, new_width) * scale_x
    updated_bbox_h = min(bbox_h, new_height) * scale_y
    updated_bbox = torch.tensor([updated_bbox_x, updated_bbox_y, updated_bbox_w, updated_bbox_h])

    # Cache the cropping info
    cropping_info = {
        'crop_left': crop_left,
        'crop_top': crop_top,
        'scale_x': scale_x,
        'scale_y': scale_y,
    }
    return cropped_image, updated_bbox, cropping_info


def is_point_inside_bbox(point, bbox):
    y, x = point
    x_min, y_min, width, height = bbox
    x_max, y_max = x_min + width, y_min + height
    return (x_min <= x <= x_max) and (y_min <= y <= y_max)


def is_blurry(image, threshold=100.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return variance < threshold


def get_similarity_heatmap(frames, video_tokens, regions, query_token, query_frame, query_bbox, query_crop,
                           response_track, encoding='region', model='dino_vitb8', save_dir='.'):
    os.makedirs(save_dir, exist_ok=True)

    if encoding == 'region':
        for instance in response_track:
            frame_number = instance['frame_number']
            frame = frames[frame_number]
            frame_tokens = video_tokens[f'frame-{frame_number}']
            frame_masks = regions[f'frame-{frame_number}']
            frame_object_tokens, frame_object_masks = [], []
            for object_info in frame_tokens:
                object_token = object_info['region_feature'][None]
                object_bbox = object_info['bbox']
                max_mask_iou = 0
                for mask in frame_masks:
                    mask_bbox = mask['bbox']
                    mask_iou = iou(mask_bbox, object_bbox)
                    if mask_iou > max_mask_iou:
                        object_mask = mask_utils.decode(mask['segmentation'])
                        max_mask_iou = mask_iou

                frame_object_tokens.append(torch.tensor(object_token))
                frame_object_masks.append(torch.tensor(object_mask))
            
            frame_object_tokens = torch.cat(frame_object_tokens, dim=0)
            x = F.normalize(frame_object_tokens, p=2, dim=1)
            y = F.normalize(query_token, p=2, dim=1)
            frame_object_scores = torch.mm(x, y.T).squeeze(dim=1)

            similarity_map = torch.zeros_like(frame_object_masks[0]).float()
            for object_mask, object_score in zip(frame_object_masks, frame_object_scores):
                similarity_map = torch.maximum(similarity_map, object_mask.float() * object_score)

            plt.imshow(query_crop)
            plt.axis('off')
            plt.savefig(f'{frame_number}-query.jpg')
            plt.clf()
            plt.imshow(frame)
            plt.axis('off')
            plt.savefig(f'{frame_number}-frame.jpg')
            plt.clf()
            plt.imshow(similarity_map.numpy(), cmap='gray')
            plt.axis('off')
            plt.savefig(f'{frame_number}-sim-rbr.jpg')
            plt.clf()

    elif encoding == 'patch':
        def extract_dino_features(model, images, batch_size=32, patch_length=8, layers=[11], bbox=None):
            assert len(layers) == 1, 'Implemented for single layer extraction only.'
            image_height, image_width = images.shape[1], images.shape[2]

            transform = T.Compose([T.ToTensor(),
                                   T.Resize((384, 512), antialias=True),
                                   lambda x: x.unsqueeze(0),
                                   CenterPadding(multiple=patch_length),
                                   T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
            
            transformed_images = []
            for image in images:
                transformed_images.append(transform(image))
            transformed_images = torch.cat(transformed_images, dim=0)
            
            features = []
            for i in range(0, transformed_images.shape[0], batch_size):
                image_batch = transformed_images[i:(i + batch_size)].to(device=device, dtype=torch.bfloat16)
                with torch.inference_mode():
                    n = 12 - layers[0]
                    features_out = model.get_intermediate_layers(image_batch, n=n)[0]
                    features_out = features_out[:, 1:].cpu()
                    features.append(features_out)
                    torch.cuda.empty_cache()
            features = torch.cat(features, dim=0)

            if bbox is not None:
                x_scale = 512 / image_width
                y_scale = 384 / image_height
                resized_bbox = [
                    bbox[0] * x_scale,
                    bbox[1] * y_scale,
                    bbox[2] * x_scale,
                    bbox[3] * y_scale
                ]
                
                num_patches_h = 384 // patch_length
                num_patches_w = 512 // patch_length
                xmin_patch = int(resized_bbox[0] // patch_length)
                ymin_patch = int(resized_bbox[1] // patch_length)
                xmax_patch = int((resized_bbox[0] + resized_bbox[2]) // patch_length)
                ymax_patch = int((resized_bbox[1] + resized_bbox[3]) // patch_length)

                xmin_patch = max(0, xmin_patch)
                ymin_patch = max(0, ymin_patch)
                xmax_patch = min(num_patches_w - 1, xmax_patch)
                ymax_patch = min(num_patches_h - 1, ymax_patch)

                token_indices = []
                for y in range(ymin_patch, ymax_patch + 1):
                    for x in range(xmin_patch, xmax_patch + 1):
                        token_index = y * num_patches_w + x
                        token_indices.append(token_index)
                token_indices = torch.tensor(token_indices).long()
                
                features = torch.index_select(features, 1, token_indices)
            return features.detach().cpu().to(torch.float32)

        # Patch-based representations
        if model == 'dino_vitb8':
            model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
            model = model.to(device=device, dtype=torch.bfloat16)
            query_patch_tokens = extract_dino_features(model, query_frame[None], bbox=query_bbox)[0]

        for instance in response_track:
            frame_number = instance['frame_number']
            frame = frames[frame_number]
            frame_patch_tokens = extract_dino_features(model, frame[None])[0]
            
            x = F.normalize(frame_patch_tokens, p=2, dim=1)
            y = F.normalize(query_patch_tokens, p=2, dim=1)
            frame_patch_scores = torch.mm(x, y.T).squeeze(dim=1)
            frame_patch_scores = torch.max(frame_patch_scores, dim=1)[0]

            similarity_map = frame_patch_scores.reshape(48, 64)
            similarity_map = F.interpolate(similarity_map[None][None], size=(frames.shape[1], frames.shape[2]),
                                           mode='bilinear', align_corners=False)[0][0]

            plt.imshow(query_crop)
            plt.axis('off')
            plt.savefig(f'{save_dir}/{frame_number}-query.jpg')
            plt.clf()
            plt.imshow(frame)
            plt.axis('off')
            plt.savefig(f'{save_dir}/{frame_number}-frame.jpg')
            plt.clf()
            plt.imshow(similarity_map.numpy(), cmap='gray')
            plt.axis('off')
            plt.savefig(f'{save_dir}/{frame_number}-sim-pbr.jpg')
            plt.clf()
    
    else:
        raise ValueError(f'{encoding}-based encoding is not implemented.')


def print_log(log_str, save_dir=None):
    print(log_str)
    if save_dir is not None:
        log_file = os.path.join(save_dir, 'log.txt')
        with open(log_file, 'a') as f:
            f.write(log_str + '\n')


def format_predictions(video_uids, clip_uids, qset_ids, predicted_response_tracks, ground_truth_response_tracks,
                       frame_dimensions):
    predictions = {
        'version': '1.0.5',
        'challenge': 'ego4d_vq2d_challenge',
        'results': {'videos': []},
    }

    def assign_annotation_uid(qset_uids):
        annotation_uids = []
        last_assigned_uid = {}
        for uid in qset_uids:
            if uid in last_assigned_uid:
                annotation_uids.append(last_assigned_uid[uid] + 1)
                last_assigned_uid[uid] += 1
            else:
                annotation_uids.append(1)
                last_assigned_uid[uid] = 1
        return annotation_uids

    for video_uid in set(video_uids):
        video_predictions = {'video_uid': video_uid, 'clips': []}
        video_clip_uids = [clip_uids[i] for i in range(len(video_uids)) if video_uids[i] == video_uid]
        video_qset_ids = [qset_ids[i] for i in range(len(video_uids)) if video_uids[i] == video_uid]
        video_predicted_tracks = [predicted_response_tracks[i]
                                  for i in range(len(video_uids)) if video_uids[i] == video_uid]
        video_ground_truth_tracks = [ground_truth_response_tracks[i]
                                     for i in range(len(video_uids)) if video_uids[i] == video_uid]
        video_frame_dimensions = [frame_dimensions[i] for i in range(len(video_uids)) if video_uids[i] == video_uid]
        assert len(set(video_frame_dimensions)) == 1
        video_predictions['frame_height'] = video_frame_dimensions[0][0]
        video_predictions['frame_width'] = video_frame_dimensions[0][1]
        
        for clip_uid in set(video_clip_uids):
            clip_predictions = {'clip_uid': clip_uid, 'predictions': []}
            clip_qset_ids = [video_qset_ids[i] for i in range(len(video_clip_uids)) if video_clip_uids[i] == clip_uid]
            clip_predicted_tracks = [video_predicted_tracks[i]
                                     for i in range(len(video_clip_uids)) if video_clip_uids[i] == clip_uid]
            clip_ground_truth_tracks = [video_ground_truth_tracks[i]
                                        for i in range(len(video_clip_uids)) if video_clip_uids[i] == clip_uid]
            clip_annotation_uids = assign_annotation_uid(clip_qset_ids)

            for annotation_uid in set(clip_annotation_uids):
                annotation_qset_ids = [clip_qset_ids[i] 
                                       for i in range(len(clip_annotation_uids))
                                       if clip_annotation_uids[i] == annotation_uid]
                annotation_predicted_tracks = [clip_predicted_tracks[i]
                                               for i in range(len(clip_annotation_uids))
                                               if clip_annotation_uids[i] == annotation_uid]
                annotation_ground_truth_tracks = [clip_ground_truth_tracks[i]
                                                  for i in range(len(clip_annotation_uids))
                                                  if clip_annotation_uids[i] == annotation_uid]
                apred = {'query_sets': {}}
                for idx, qset_id in enumerate(annotation_qset_ids):
                    apred['query_sets'][qset_id] = annotation_predicted_tracks[idx][0].to_json()
                    if annotation_ground_truth_tracks[idx].bboxes[0].fno != -1:
                        apred['query_sets'][qset_id + '-gt'] = annotation_ground_truth_tracks[idx].to_json()
                clip_predictions['predictions'].append(apred)
            video_predictions['clips'].append(clip_predictions)
        predictions['results']['videos'].append(video_predictions)

    return predictions