import os
import numpy as np
from matplotlib import pyplot as plt
from skimage.segmentation import slic as sk_slic
from scipy import ndimage as ndi
import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from model import FeatureExtractor, RegionEncoder, TokenAggregator


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class REN(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Keep a reference
        self.config = config
        ren_cfg = config['ren']

        # ----- Device -----
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ----- Params & Architecture (with safe defaults) -----
        params = ren_cfg.get('parameters', {})
        arch   = ren_cfg.get('architecture', {})
        pretr  = ren_cfg.get('pretrained', {})
        logcfg = ren_cfg.get('logging', {})

        self.extractor_name   = params.get('feature_extractor', 'dino_vitb8')
        self.image_resolution = int(params.get('image_resolution', 256))
        self.grid_size        = int(params.get('grid_size', 32))
        self.use_slic         = bool(params.get('use_slic', False))
        self.aggregate_tokens = bool(params.get('aggregate_tokens', False))
        self.token_variant    = str(params.get('token_variant', 'ren'))  # 'ren' | 'aligned_ren'
        self.patch_size       = int(pretr.get('patch_sizes', [8])[0])

        # ----- Submodules -----
        self.feature_extractor = FeatureExtractor(ren_cfg, device=self.device)
        self.region_encoder    = RegionEncoder(ren_cfg).to(self.device).eval()

        # Token aggregator is only required when aggregate_tokens=True
        try:
            if self.aggregate_tokens:
                self.token_aggregator = TokenAggregator(ren_cfg)
            else:
                self.token_aggregator = None
        except NameError:
            # If TokenAggregator class is not imported/available, make it optional
            self.token_aggregator = None
            if self.aggregate_tokens:
                print("[REN][warn] aggregate_tokens=True but TokenAggregator is unavailable; disabling aggregation.")

        # ----- Checkpoint -----
        self.checkpoint_path = params.get('ren_ckpt', os.path.join(
            logcfg.get('save_dir', 'logs/'),
            logcfg.get('exp_name', 'ren-dino-vitb8'),
            'checkpoint.pth'
        ))
        self.load_checkpoint()  # uses CPU map_location

        # ----- Grid points in model image space (S x S) -----
        # integer grid in [1, S-2] to avoid border effects
        S = self.image_resolution
        x_coords = np.linspace(1, S - 2, self.grid_size, dtype=int)
        y_coords = np.linspace(1, S - 2, self.grid_size, dtype=int)
        self.grid_points = torch.tensor(
            [(y, x) for y in y_coords for x in x_coords],
            dtype=torch.int32
        )

    def load_checkpoint(self):
        ckpt_path = self.config['ren']['parameters']['ren_ckpt']
        print(ckpt_path)
        ckpt_path = os.path.abspath(ckpt_path)
        print(f"[REN] Trying checkpoint: {ckpt_path}")

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"[REN] checkpoint not found: {ckpt_path}")

        size = os.path.getsize(ckpt_path)
        print(f"[REN] checkpoint size: {size} bytes")
        if size < 1_000_000:
            print(ckpt_path)
            raise RuntimeError(f"[REN] checkpoint too small: {size} bytes (wrong file)")

        with open(ckpt_path, "rb") as f:
            head = f.read(128)
        # Basic sanity checks
        if b"version https://git-lfs.github.com/spec" in head:
            raise RuntimeError("[REN] This file is a Git LFS pointer, not weights.")
        if head[:2] == b"PK":
            print("[REN] zip-like Torch checkpoint (ok)")
        else:
            # many torch checkpoints are zip (PK). Some are legacy pickles; both can work,
            # but your error indicates something unreadable, so warn loudly here.
            print("[REN] Non-zip header; will still try torch.load")

        # Now actually load, on CPU
        print("[REN] Loading weights with torch.load(...)")
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        # Accept both formats: whole dict or nested state
        state = checkpoint.get("region_encoder_state", checkpoint)
        missing, unexpected = self.region_encoder.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[REN] Non-strict load -> missing={len(missing)} unexpected={len(unexpected)}")
        self.checkpoint_path = ckpt_path
        print(f"[REN] Loaded checkpoint from: {self.checkpoint_path}")


# add once at top of ren.py if not present:
# from skimage.segmentation import slic
# import numpy as np
# import torch

    def get_slic_points(self, images, num_segments):
        """
        images: either
        - np.ndarray [T,H,W,3] (uint8 or float in [0,1]/[0,255])
        - torch.Tensor [T,H,W,3] or [T,3,H,W] (uint8/float)
        Returns:
        prompts:     list[Tensor] length T, each Tensor [K,2] with (y,x) centers (int32)
        superpixels: list[np.ndarray] length T, each [H,W] int labels
        """
        prompts = []
        superpixels = []

        # normalize to numpy HWC uint8 per frame for SLIC
        if isinstance(images, torch.Tensor):
            t = images
            if t.ndim != 4:
                raise ValueError(f"get_slic_points expects 4D, got {tuple(t.shape)}")
            if t.shape[-1] == 3:       # [T,H,W,3]
                t_hwc = t
            elif t.shape[1] == 3:      # [T,3,H,W] -> [T,H,W,3]
                t_hwc = t.permute(0, 2, 3, 1)
            else:
                raise ValueError(f"Ambiguous channel dim in {tuple(t.shape)}")
            if t_hwc.is_floating_point():
                # assume [0,1] or [0,255]; bring to uint8
                t_hwc = (t_hwc.clamp(0, 1) * 255.0).to(torch.uint8)
            imgs_hwc = t_hwc.detach().cpu().numpy()
        elif isinstance(images, np.ndarray):
            if images.ndim != 4 or images.shape[-1] != 3:
                raise ValueError(f"Expected numpy [T,H,W,3], got {images.shape}")
            imgs_hwc = images
            if imgs_hwc.dtype != np.uint8:
                # assume float [0,1] or [0,255]; convert to [0,255] uint8
                if imgs_hwc.max() <= 1.0:
                    imgs_hwc = (imgs_hwc * 255.0).astype(np.uint8)
                else:
                    imgs_hwc = np.clip(imgs_hwc, 0, 255).astype(np.uint8)
        else:
            raise TypeError(f"Unsupported images type: {type(images)}")

        # run SLIC per frame and compute (y,x) centers for each superpixel
        for fi in range(imgs_hwc.shape[0]):
            image = imgs_hwc[fi]  # [H,W,3] uint8
            H, W = image.shape[:2]

            # skimage.slic expects float in [0,1]
            img_float = image.astype(np.float32) / 255.0

            # produce approximately num_segments superpixels
            segments = slic(
                img_float,
                n_segments=int(num_segments),
                compactness=10.0,
                start_label=0,
                channel_axis=-1,
            ).astype(np.int32)  # [H,W]

            # compute centers (y,x) per label
            K = segments.max() + 1
            cyx = []
            for lab in range(K):
                ys, xs = np.where(segments == lab)
                if ys.size == 0:
                    continue
                cyx.append([int(ys.mean()), int(xs.mean())])

            if len(cyx) == 0:
                # fallback: center of frame
                cyx = [[H // 2, W // 2]]

            centers = torch.tensor(cyx, dtype=torch.int32)

            # pad/truncate to exactly num_segments (your downstream code expects fixed count)
            target = int(num_segments)
            if centers.shape[0] < target:
                pad_len = target - centers.shape[0]
                pad_row = centers[-1].unsqueeze(0).repeat(pad_len, 1)
                centers = torch.cat([centers, pad_row], dim=0)
            elif centers.shape[0] > target:
                centers = centers[:target]

            prompts.append(centers)      # [num_segments, 2] (y,x) int32
            superpixels.append(segments) # [H,W] int32

        return prompts, superpixels


    def forward(self,images,matching_tokens=None,matching_threshold=0.0,batch_size=4,key_prefix='frame',cache_attn_maps=False,):
        """
        Compatible with the older caller that expects:
        enc = REN(...); image_tokens = enc(frames, matching_tokens, matching_threshold, batch_size, key_prefix, cache_attn_maps)
        Returns:
        dict mapping f'{key_prefix}-{i}' -> list of { 'point': [y,x], 'region_feature': np.ndarray, (optional) 'attn_map': np.ndarray }
        Notes:
        - No SLIC. Points come from self.grid_points in REN image space mapped back to original (H,W).
        - If matching_tokens is given (Tensor [Q,D]), keeps only tokens with cosine >= matching_threshold
            (or the best one if none reach threshold).
        """
        import numpy as np
        import torch
        import torch.nn.functional as F

        # ---- normalize inputs to torch BCHW float32 in [0,1] AND keep HWC uint8 copy for geometry ----
        # Build an HWC numpy copy for geometry (height/width) and to later map points back
        if isinstance(images, torch.Tensor):
            t_any = images
            if t_any.ndim != 4:
                raise ValueError(f"REN.forward expects 4D input, got {tuple(t_any.shape)}")
            if t_any.shape[-1] == 3:  # [T,H,W,3]
                t_hwc = t_any
            elif t_any.shape[1] == 3:  # [T,3,H,W] -> [T,H,W,3]
                t_hwc = t_any.permute(0, 2, 3, 1)
            else:
                raise ValueError(f"Ambiguous channel dim in {tuple(t_any.shape)}")
            if t_hwc.is_floating_point():
                t_u8 = (t_hwc.clamp(0, 1) * 255.0).to(torch.uint8)
            else:
                t_u8 = t_hwc.to(torch.uint8)
            hwc_np = t_u8.detach().cpu().numpy()
        elif isinstance(images, np.ndarray):
            if images.ndim != 4 or images.shape[-1] != 3:
                raise ValueError(f"Expected numpy [T,H,W,3], got {images.shape}")
            hwc_np = images
            if hwc_np.dtype != np.uint8:
                if hwc_np.max() <= 1.0:
                    hwc_np = (hwc_np * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    hwc_np = hwc_np.clip(0, 255).astype(np.uint8)
        else:
            raise TypeError(f"Unsupported images type: {type(images)}")

        T = hwc_np.shape[0]
        H0, W0 = hwc_np.shape[1], hwc_np.shape[2]

        # Build BCHW float32 in [0,1] at REN resolution
        t = torch.from_numpy(hwc_np)  # [T,H,W,3] uint8
        t = t.permute(0, 3, 1, 2).float() / 255.0  # [T,3,H,W] float
        S = self.image_resolution
        if t.shape[2] != S or t.shape[3] != S:
            t = F.interpolate(t, size=(S, S), mode="bilinear", align_corners=False)

        t = t.to(self.device, non_blocking=True)

        # ---- feature extraction on resized BCHW ----
        image_tokens = {}  # output dict
        with torch.no_grad():
            for i0 in range(0, T, batch_size):
                i1 = min(T, i0 + batch_size)
                image_batch = t[i0:i1]  # [B,3,S,S]
                # returns (feature_tokens, feature_maps, cls_tokens) but we only need feature_maps
                _, feature_maps = self.feature_extractor(self.extractor_name, image_batch, resize=False)

                # region encoder on model grid points
                # prompts: repeat self.grid_points per image in batch
                prompts = [self.grid_points for _ in range(image_batch.shape[0])]
                ren_out = self.region_encoder(feature_maps, prompts)
                # select variant
                if self.aggregate_tokens and self.token_aggregator is not None:
                    ag = self.token_aggregator(
                        ren_out['pred_tokens'],
                        ren_out['proj_tokens'],
                        ren_out['attn_scores'][-1],
                        prompts
                    )
                    region_tokens = ag['aggregated_pred_tokens'] if self.token_variant == 'ren' else ag['aggregated_proj_tokens']
                else:
                    region_tokens = ren_out['pred_tokens'] if self.token_variant == 'ren' else ren_out['proj_tokens']
                # region_tokens: [B,G,D] (assumed)

                # (Optional) filter by matching tokens
                # matching_tokens: [Q,D]
                keep_mask_list = []
                if matching_tokens is not None:
                    x = torch.nn.functional.normalize(region_tokens, p=2, dim=-1)  # [B,G,D]
                    y = torch.nn.functional.normalize(matching_tokens.to(region_tokens.device), p=2, dim=-1)  # [Q,D]
                    # cosine per (B,G) vs (Q)
                    cos = torch.einsum('bgd,qd->bgq', x, y)  # [B,G,Q]
                    cos_max, _ = cos.max(dim=-1)            # [B,G]
                    # keep >= threshold; if none kept, keep best one
                    for b in range(cos_max.shape[0]):
                        scores_b = cos_max[b]  # [G]
                        mask_b = scores_b >= matching_threshold
                        if not mask_b.any():
                            argm = int(scores_b.argmax())
                            mask = torch.zeros_like(scores_b, dtype=torch.bool)
                            mask[argm] = True
                        else:
                            mask = mask_b
                        keep_mask_list.append(mask)
                else:
                    # keep all
                    keep_mask_list = [torch.ones(region_tokens.shape[1], dtype=torch.bool, device=region_tokens.device)
                                    for _ in range(region_tokens.shape[0])]

                # map model-space grid points back to ORIGINAL (H0,W0)
                gp = self.grid_points.to(torch.float32)      # [G,2] (y,x) in SxS space
                scale_y = float(H0) / float(S)
                scale_x = float(W0) / float(S)
                pts_xy = gp.clone()
                pts_xy[:, 0] = pts_xy[:, 0] * scale_y  # y
                pts_xy[:, 1] = pts_xy[:, 1] * scale_x  # x

                # build output dict
                for bi in range(i0, i1):
                    b = bi - i0
                    key = f"{key_prefix}-{bi}"
                    feats_b = region_tokens[b]                    # [G,D]
                    keep_b  = keep_mask_list[b]                   # [G]
                    pts_b   = pts_xy.to(region_tokens.device)     # [G,2]
                    feats_b = feats_b[keep_b].detach().cpu().numpy()
                    pts_b   = pts_b[keep_b].detach().cpu().numpy()

                    image_tokens[key] = []
                    for j in range(feats_b.shape[0]):
                        # point as [y,x] in ORIGINAL (H0,W0)
                        py = int(round(pts_b[j, 0].item()))
                        px = int(round(pts_b[j, 1].item()))
                        image_tokens[key].append({
                            'point': [py, px],
                            'region_feature': feats_b[j],
                            # 'attn_map': ...  # add if you wire attention maps later
                        })

                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return image_tokens


    

class XREN(nn.Module):
    def __init__(self, config):
        super(XREN, self).__init__()

        # Instantiate REN modules
        self.ren_extractor_name = config['parameters']['ren_feature_extractor']
        self.target_extractor_name = config['parameters']['target_feature_extractor']
        self.feature_extractor = FeatureExtractor(config['ren'], device=device)
        self.region_encoder = RegionEncoder(config['ren']).to(device).eval()
        self.token_aggregator = TokenAggregator(config['ren'])

        # Instantiate target models
        self.target_extractor_name = config['parameters']['target_feature_extractor']
        if self.target_extractor_name == 'siglip_vitg16':
            self.target_model = AutoModel.from_pretrained('google/siglip2-giant-opt-patch16-384',
                                                          device_map=device).vision_model

        # Load the checkpoint
        self.checkpoint_path = config['parameters']['ren_ckpt']
        self.load_checkpoint()

        # Grid points for region tokens
        self.image_resolution = config['parameters']['image_resolution']
        self.grid_size = config['parameters']['grid_size']
        x_coords = np.linspace(1, self.image_resolution - 2, self.grid_size, dtype=int)
        y_coords = np.linspace(1, self.image_resolution - 2, self.grid_size, dtype=int)
        self.grid_points = torch.tensor([(y, x) for y in y_coords for x in x_coords])

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path,map_location=torch.device("cpu"))
            self.start_epoch = checkpoint['epoch']
            self.start_iter = checkpoint['iter_count']
            self.region_encoder.load_state_dict(checkpoint['region_encoder_state'])
            print(f'Checkpoint loaded from epoch {self.start_epoch}, iteration {self.start_iter}.')
        else:
            print('No checkpoint found; exiting..')
            exit()

    def get_slic_points(self, images, num_segments):
        prompts, superpixels = [], []
        for image in images:
            image = (image.permute(1, 2, 0).cpu().numpy().copy() * 255).astype(np.uint8)

            # Get SLIC superpixels
            slic = slic(num_components=num_segments, compactness=256)
            segments = slic.iterate(image)
            slic_segments = segments.max() + 1

            # Get center of mass for all segments
            centers = np.array(ndi.center_of_mass(np.ones_like(segments), labels=segments, index=np.arange(slic_segments)))
            centers = np.round(centers).astype(int)

            centers[:, 0] = np.clip(centers[:, 0], 0, segments.shape[0] - 1)
            centers[:, 1] = np.clip(centers[:, 1], 0, segments.shape[1] - 1)

            # Check which centers are outside their own superpixel
            valid = segments[centers[:, 0], centers[:, 1]] == np.arange(slic_segments)

            # For invalid centers, pick a pixel inside the superpixel
            if not np.all(valid):
                for seg_id in np.where(~valid)[0]:
                    mask = (segments == seg_id)
                    yx = np.argwhere(mask)
                    if len(yx) > 0:
                        centers[seg_id] = yx[len(yx) // 2]
            centers = torch.tensor(centers, dtype=torch.int64)

            # Pad if needed
            pad_len = num_segments - len(centers)
            if pad_len > 0:
                center_padding = torch.stack([centers[-1]] * pad_len)
                centers = torch.cat([centers, center_padding], dim=0)

            prompts.append(centers)
            superpixels.append(segments)
        return prompts, superpixels

    def get_pooled_tokens(self, images, region_masks, batch_size=1024):
        if self.target_extractor_name == 'siglip_vitg16':
            transform = kornia.augmentation.AugmentationSequential(
                kornia.augmentation.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            )
            transformed_images = transform(images)
            patch_length = 16

            for i in range(0, transformed_images.shape[0], batch_size):
                image_batch = transformed_images[i:(i + batch_size)].to(device=images.device)
                with torch.inference_mode():
                    features_out = self.target_model(pixel_values=image_batch, output_hidden_states=False,
                                                     output_attentions=False).last_hidden_state
                    pooled_tokens = []
                    for image_idx in range(features_out.shape[0]):
                        pooled_tokens.append([])
                        image_features = features_out[image_idx]
                        mask_resolution = self.image_resolution // patch_length
                        masks = F.interpolate(region_masks[image_idx].unsqueeze(1), size=(mask_resolution, mask_resolution),
                                              mode='nearest').squeeze(1)
                        for mask in masks:
                            mask_flatten = mask.view(-1)
                            sampled_image_features = image_features[mask_flatten]
                            pooled_token = self.target_model.head(sampled_image_features.unsqueeze(0))
                            pooled_tokens[-1].append(pooled_token)
                        pooled_tokens[-1] = torch.cat(pooled_tokens[-1])
        return pooled_tokens
    
    def forward(self, images, visualize_mask=False):
        _, feature_maps = self.feature_extractor(self.ren_extractor_name, images, resize=False)
        prompts, segments = self.get_slic_points(images, self.grid_size * self.grid_size)
        ren_outputs = self.region_encoder(feature_maps, prompts)
        aggregated_outputs = self.token_aggregator(ren_outputs['pred_tokens'], ren_outputs['proj_tokens'],
                                                   ren_outputs['attn_scores'][-1], prompts)
        all_grouped_points = aggregated_outputs['all_grouped_points']
        
        # Create region masks
        region_masks = []
        for batch_idx in range(len(all_grouped_points)):
            region_masks.append([])
            segment = segments[batch_idx]
            for group_idx in range(len(all_grouped_points[batch_idx])):
                group_points = all_grouped_points[batch_idx][group_idx]
                mask = np.zeros_like(segment, dtype=bool)
                for yx in group_points:
                    y, x = yx
                    sp_label = segment[y, x]
                    mask |= (segment == sp_label)
                region_masks[batch_idx].append(torch.tensor(mask.astype(np.uint8)))

                if visualize_mask:
                    plt.imshow(mask[..., None] * images[batch_idx].permute(1, 2, 0).cpu().numpy())
                    plt.savefig(f'{batch_idx}-{group_idx}.jpg')
                    plt.imshow(images[batch_idx].permute(1, 2, 0).cpu().numpy())
                    plt.savefig(f'{batch_idx}-image.jpg')
        region_masks = [torch.stack(masks).to(device) for masks in region_masks]

        # Pool features of the target model
        pooled_tokens = self.get_pooled_tokens(images, region_masks)
        return pooled_tokens