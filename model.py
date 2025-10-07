import sys
import math
import numpy as np
import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from task_utils import CenterPadding, upsample_features, group_predictions

sys.path.append('segment_anything/')
from segment_anything.sam2.build_sam import build_sam2
from segment_anything.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


class FeatureExtractor():
    def __init__(self, config, device, return_class_token=False):
        self.device = device
        self.models = {}
        self.models['dino_vitb8'] = torch.hub.load('facebookresearch/dino:main',
                                                   'dino_vitb8').to(device)
        self.models['dinov2_vitl14'] = torch.hub.load('facebookresearch/dinov2',
                                                      'dinov2_vitl14').to(device)
        self.models['openclip_vitg14'] = open_clip.create_model('ViT-g-14',
                                                                pretrained='laion2b_s34b_b88k',
                                                                device=device).visual
        self.return_class_token = return_class_token
    
    def extract_dino(self, model, images, batch_size=1024, patch_length=8, layers=[11]):
        transform = kornia.augmentation.AugmentationSequential(
            CenterPadding(multiple=patch_length),
            kornia.augmentation.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        )
        transformed_images = transform(images)
        
        feature_tokens, feature_maps, cls_tokens = [], [], []
        for i in range(0, transformed_images.shape[0], batch_size):
            image_batch = transformed_images[i:(i + batch_size)].to(device=self.device)
            with torch.inference_mode():
                n = 12 - layers[0]
                features_out = model.get_intermediate_layers(image_batch, n=n)[0]
                cls_token = features_out[:, 0]
                features_out = features_out[:, 1:]
                cls_tokens.append(cls_token)
                feature_tokens.append(features_out)

                B, _, C = features_out.size()
                H, W = image_batch.shape[2], image_batch.shape[3]
                patch_H, patch_W = math.ceil(H / patch_length), math.ceil(W / patch_length)
                features_out = features_out.permute(0, 2, 1).view(B, C, patch_H, patch_W)
                feature_maps.append(features_out)
        feature_tokens = torch.cat(feature_tokens, dim=0)
        feature_maps = torch.cat(feature_maps, dim=0)
        cls_tokens = torch.cat(cls_tokens, dim=0)
        return feature_tokens, feature_maps, cls_tokens

    def extract_dinov2(self, model, images, batch_size=1024, patch_length=14, layers=[23]):
        transform = kornia.augmentation.AugmentationSequential(
            CenterPadding(multiple=patch_length),
            kornia.augmentation.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        )
        transformed_images = transform(images)
        
        feature_tokens, feature_maps, cls_tokens = [], [], []
        for i in range(0, transformed_images.shape[0], batch_size):
            image_batch = transformed_images[i:(i + batch_size)].to(device=self.device)
            with torch.inference_mode():
                features_out = model.get_intermediate_layers(image_batch, return_class_token=True, n=layers)[0]
                cls_token = features_out[1]
                features_out = features_out[0]
                cls_tokens.append(cls_token)
                feature_tokens.append(features_out)

                B, _, C = features_out.size()
                H, W = image_batch.shape[2], image_batch.shape[3]
                patch_H, patch_W = math.ceil(H / patch_length), math.ceil(W / patch_length)
                features_out = features_out.permute(0, 2, 1).view(B, C, patch_H, patch_W)
                feature_maps.append(features_out)
        feature_tokens = torch.cat(feature_tokens, dim=0)
        feature_maps = torch.cat(feature_maps, dim=0)
        cls_tokens = torch.cat(cls_tokens, dim=0)
        return feature_tokens, feature_maps, cls_tokens
    
    def extract_openclip(self, model, images, batch_size=1024, patch_length=14):
        transform = kornia.augmentation.AugmentationSequential(
            CenterPadding(multiple=patch_length),
            kornia.augmentation.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        )
        transformed_images = transform(images)
        
        feature_tokens, feature_maps, cls_tokens = [], [], []
        for i in range(0, transformed_images.shape[0], batch_size):
            image_batch = transformed_images[i:(i + batch_size)].to(device=self.device)
            with torch.inference_mode():
                features_out = model.forward_intermediates(image_batch, indices=1, intermediates_only=False)
                cls_token = features_out['image_features']
                features_out = features_out['image_intermediates'][-1]
                cls_tokens.append(cls_token)
                feature_maps.append(features_out)
                feature_tokens.append(features_out.flatten(2).permute(0, 2, 1))
        feature_tokens = torch.cat(feature_tokens, dim=0)
        feature_maps = torch.cat(feature_maps, dim=0)
        cls_tokens = torch.cat(cls_tokens, dim=0)
        return feature_tokens, feature_maps, cls_tokens
        
    def __call__(self, model, images, resize=True):
        if model == 'dino_vitb8':
            feature_tokens, feature_maps, cls_tokens = self.extract_dino(self.models[model], images)
            patch_size = 8
        elif model == 'dinov2_vitl14':
            feature_tokens, feature_maps, cls_tokens = self.extract_dinov2(self.models[model], images)
            patch_size = 14
        elif model == 'openclip_vitg14':
            feature_tokens, feature_maps, cls_tokens = self.extract_openclip(self.models[model], images)
            patch_size = 14
        else:
            raise ValueError(f'Feature extraction is not implemented for {model}.')
        
        if resize:
            image_height, image_width = images.shape[2], images.shape[3]
            padded_height = math.ceil(image_height / patch_size) * patch_size
            padded_width = math.ceil(image_width / patch_size) * patch_size
            resized_feature_maps = []
            chunk_size = 32
            for i in range(0, len(feature_maps), chunk_size):
                resized_feature_maps.append(upsample_features(feature_maps[i:i + chunk_size], image_height,
                                                              image_width, padded_height, padded_width))
            feature_maps = torch.cat(resized_feature_maps)
        
        if self.return_class_token:
            return feature_tokens, feature_maps, cls_tokens
        return feature_tokens, feature_maps


class RegionExtractor():
    def __init__(self, config, device):
        self.device = device
        self.region_extractor = config['pretrained']['region_extractor']
        sam2 = build_sam2(config['pretrained']['sam2_hieral_config'], config['pretrained']['sam2_hieral_ckpt'],
                            device=device, apply_postprocessing=False)
        self.mask_generator = SAM2AutomaticMaskGenerator(sam2, stability_score_thresh=0.95)

    def __call__(self, images):
        regions = []

        if self.region_extractor == 'batched_sam':
            images = (self.preprocess(images) * 255).to(device=self.device, dtype=torch.uint8)
            batched_input = []
            for image in images:
                batched_input.append({
                    'image': image,
                    'point_coords': self.input_points,
                    'point_labels': self.input_labels,
                    'original_size': image.shape[1:]
                })

            segmentations = self.sam.individual_forward(batched_input, multimask_output=True)
            for image_masks in segmentations:
                regions.append([])
                for mask in image_masks:
                    regions[-1].append(self.postprocess(mask[None])[0].cpu().numpy())

        else:
            images = images.permute(0, 2, 3, 1).numpy()
            for image in images:
                image_masks = self.mask_generator.generate(image)
                image_regions = []
                for mask in image_masks:
                    image_regions.append(mask['segmentation'])
                regions.append(image_regions)
        
        regions = [torch.tensor(np.array(r)) for r in regions]
        return regions
    

class RegionTokensGenerator():
    def __init__(self, pooling_method='average', device='cuda'):
        self.pooling_method = pooling_method
        self.device = device

    def __call__(self, features, regions):
        region_tokens = []
        for image_features, image_regions in zip(features, regions):
            image_features = image_features.to(self.device)
            image_regions = image_regions.to(self.device)
            if image_regions.numel() == 0:
                region_tokens.append(torch.zeros((0, image_features.shape[0]), device=self.device))
                continue
            region_features = torch.einsum('rhw,chw->rc', image_regions.float(), image_features)
            if self.pooling_method == 'average':
                valid_elements = image_regions.sum(dim=(1, 2), dtype=torch.float32).clamp(min=1).unsqueeze(1)
                region_features = region_features / valid_elements
            region_tokens.append(region_features)
        return region_tokens


class FeatureProjector(nn.Module):
    def __init__(self, hidden_dim, feature_extractors, use_bias=False):
        super(FeatureProjector, self).__init__()
        self.proj = nn.ModuleDict()
        for extractor_name in feature_extractors:
            if extractor_name == 'dino_vitb8':
                self.proj['dino_vitb8'] = nn.Linear(768, hidden_dim, bias=use_bias)
                nn.init.kaiming_normal_(self.proj['dino_vitb8'].weight, mode='fan_in', nonlinearity='linear')
                if use_bias:
                    nn.init.zeros_(self.proj['dino_vitb8'].bias)
            elif extractor_name == 'dinov2_vitl14':
                self.proj['dinov2_vitl14'] = nn.Linear(1024, hidden_dim, bias=use_bias)
                nn.init.kaiming_normal_(self.proj['dinov2_vitl14'].weight, mode='fan_in', nonlinearity='linear')
                if use_bias:
                    nn.init.zeros_(self.proj['dinov2_vitl14'].bias)
            else:
                raise ValueError(f"Unsupported feature extractor: {extractor_name}")
    
    def forward(self, model, features):
        if model not in self.proj:
            raise ValueError(f'No projection implemented for {model}')
        out = self.proj[model](features)
        return out


class PositionalEmbedding2D(nn.Module):
    def __init__(self, embedding_dim=64, scale=None):
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        generator = torch.Generator()
        generator.manual_seed(42)
        self.register_buffer("positional_encoding_gaussian_matrix", 
                             scale * torch.randn((2, embedding_dim // 2), generator=generator))

    def _pe_encoding(self, coords):
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size):
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)


class AttentionLayer(nn.Module):
    def __init__(self, q_dim, kv_dim, hidden_dim, num_heads=8, dropout=0.1, use_bias=False):
        super(AttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0, 'Hidden dimension must be a multiple of the number of heads.'
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(q_dim, hidden_dim)
        self.k_proj = nn.Linear(kv_dim, hidden_dim)
        self.v_proj = nn.Linear(kv_dim, hidden_dim)
        nn.init.kaiming_normal_(self.q_proj.weight, mode='fan_in', nonlinearity='linear')
        nn.init.kaiming_normal_(self.k_proj.weight, mode='fan_in', nonlinearity='linear')
        nn.init.kaiming_normal_(self.v_proj.weight, mode='fan_in', nonlinearity='linear')
        if use_bias:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)

        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.kaiming_normal_(self.out_proj.weight, mode='fan_in', nonlinearity='linear')
        if use_bias:
            nn.init.zeros_(self.out_proj.bias)

        self.scale = (hidden_dim // num_heads) ** -0.5
    
    def forward(self, q, k, v, mask=None, project_values=True, attention_threshold=None):
        batch_size, q_len, _ = q.shape
        _, kv_len, _ = k.shape

        query = self.q_proj(q).view(batch_size, q_len, self.num_heads, -1).transpose(1, 2)
        key = self.k_proj(k).view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)
        if project_values:
            value = self.v_proj(v).view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)
        else:
            value = v.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)

        query = self.q_norm(query)
        key = self.k_norm(key)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        if attention_threshold is not None:
            max_attn_scores, _ = attn_scores.max(dim=-1, keepdim=True)
            thresholding_mask = attn_scores >= (attention_threshold * max_attn_scores)
            attn_scores = attn_scores.masked_fill(thresholding_mask == 0, -1e5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, value)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, q_len, self.hidden_dim)

        out = self.out_proj(attn_out)
        return out, attn_scores


class MLPBlock(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim, dropout=0.1):
        super(MLPBlock, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, intermediate_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(intermediate_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        z = self.linear1(x)
        z = self.gelu(z)
        z = self.dropout(z)
        z = self.linear2(z)
        return z


class CrossAttentionBlock(nn.Module):
    def __init__(self, q_dim, kv_dim, hidden_dim, mlp_dim, num_heads, dropout, use_bias):
        super(CrossAttentionBlock, self).__init__()
        self.query_norm = nn.LayerNorm(q_dim)
        self.cross_attn = AttentionLayer(q_dim, kv_dim, hidden_dim, num_heads, dropout, use_bias)
        self.dropout = nn.Dropout(dropout)
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, context, mask=None, project_values=True):
        x = self.query_norm(query)
        x, attn_scores = self.cross_attn(q=x, k=context, v=context, mask=mask, project_values=project_values)
        x = self.dropout(x)
        x = x + query

        y = self.mlp_norm(x)
        y = self.mlp(y)
        out = self.out_norm(y) + x
        return out, attn_scores


class RegionEncoder(nn.Module):
    def __init__(self, config):
        super(RegionEncoder, self).__init__()
        hidden_dim = config['architecture']['hidden_dim']
        image_resolution = config['parameters']['image_resolution']
        upsample_features = config['parameters']['upsample_features']
        patch_size = config['pretrained']['patch_sizes'][0]

        # Create position embeddings for the prompts and feature maps
        position_embedder = PositionalEmbedding2D(hidden_dim)
        if upsample_features:
            self.location_embeddings = position_embedder((image_resolution, image_resolution))
            self.feature_embeddings = self.location_embeddings.flatten(-2).permute(1, 0).cuda()
        else:
            self.location_embeddings = position_embedder((image_resolution, image_resolution))
            self.feature_embeddings = position_embedder((image_resolution // patch_size,
                                                         image_resolution // patch_size)).flatten(-2).permute(1, 0).cuda()

        # Instantiate prompt and feature projectors
        self.prompt_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.kaiming_normal_(self.prompt_proj.weight, mode='fan_in', nonlinearity='linear')

        # Instantiate the cross-attention blocks
        self.decoder_layers = config['architecture']['decoder_layers']
        self.region_attention_layers = nn.ModuleList([
            CrossAttentionBlock(
                q_dim=hidden_dim,
                kv_dim=hidden_dim,
                hidden_dim=hidden_dim,
                mlp_dim=2 * hidden_dim,
                num_heads=config['architecture']['num_attention_heads'],
                dropout=0.1,
                use_bias=False,
            ) for _ in range(self.decoder_layers)
        ])

        # Instantiate the output projector
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.kaiming_normal_(self.out_proj.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, feature_maps, grid_points):
        feature_tokens = feature_maps.flatten(-2).permute(0, 2, 1)
        kv = feature_tokens + self.feature_embeddings[None]

        batch_size = feature_maps.shape[0]
        prompt_embeddings = [torch.stack([self.location_embeddings[:, point[0], point[1]] for point in grid_points[i]])
                             for i in range(batch_size)]
        prompt_embeddings = torch.stack(prompt_embeddings).cuda()
        q = self.prompt_proj(prompt_embeddings)
        all_attn_scores = []
        for layer_idx, layer in enumerate(self.region_attention_layers):
            q += prompt_embeddings
            if layer_idx == self.decoder_layers - 1:
                pred_tokens, attn_scores = layer(q, kv, project_values=False)
            else:
                q, attn_scores = layer(q, kv)
            all_attn_scores.append(attn_scores)
        
        proj_tokens = self.out_norm(pred_tokens)
        proj_tokens = self.out_proj(proj_tokens)
        return {
            'pred_tokens': pred_tokens,
            'proj_tokens': proj_tokens,
            'attn_scores': all_attn_scores,
        }
    

class TokenAggregator(nn.Module):
    def __init__(self, config):
        super(TokenAggregator, self).__init__()
        self.merge_similarity = config['parameters']['merge_similarity']

    def get_central_point(self, points):
        center = points.float().mean(dim=0, keepdim=True)
        dists = torch.norm(points.float() - center, dim=1)
        return points[dists.argmin()]

    def forward(self, pred_tokens, proj_tokens, attn_scores, grid_points):
        batch_size = attn_scores.size(0)
        aggregated_pred_tokens, aggregated_proj_tokens, aggregated_attn_scores, aggregated_grid_points = [], [], [], []
        all_grouped_points = []
        for batch_idx in range(batch_size):
            groups = group_predictions(pred_tokens[batch_idx], self.merge_similarity)
            batch_pred_tokens = pred_tokens[batch_idx]
            batch_proj_tokens = proj_tokens[batch_idx]
            batch_attn_scores = attn_scores[batch_idx]
            batch_grid_points = grid_points[batch_idx].to(pred_tokens.device)
            
            new_pred_tokens, new_proj_tokens, new_attn_scores, new_grid_points, grouped_points = [], [], [], [], []
            for group in groups:
                group_tensor = torch.tensor(group, device=batch_pred_tokens.device)
                group_pred_tokens = batch_pred_tokens[group_tensor]
                new_pred_tokens.append(group_pred_tokens.mean(dim=0))
                group_proj_tokens = batch_proj_tokens[group_tensor]
                new_proj_tokens.append(group_proj_tokens.mean(dim=0))
                group_attn_scores = batch_attn_scores[:, group_tensor]
                new_attn_scores.append(group_attn_scores.mean(dim=1))
                group_grid_points = batch_grid_points[group_tensor]
                new_grid_points.append(self.get_central_point(group_grid_points))
                grouped_points.append(group_grid_points)
            
            aggregated_pred_tokens.append(torch.stack(new_pred_tokens, dim=0))
            aggregated_proj_tokens.append(torch.stack(new_proj_tokens, dim=0))
            aggregated_attn_scores.append(torch.stack(new_attn_scores, dim=1))
            aggregated_grid_points.append(torch.stack(new_grid_points, dim=0))
            all_grouped_points.append(grouped_points)
        return {
            'aggregated_pred_tokens': aggregated_pred_tokens,
            'aggregated_proj_tokens': aggregated_proj_tokens,
            'aggregated_attn_scores': aggregated_attn_scores,
            'aggregated_grid_points': aggregated_grid_points,
            'all_grouped_points': all_grouped_points,
        }