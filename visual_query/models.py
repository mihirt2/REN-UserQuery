import os
import sys
import pickle
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from vq_utils import crop_using_bbox, mask_to_bbox, point_to_bbox, sliding_window_cropping, \
    generate_token_from_bbox, get_sam_pooled_tokens, get_cropping_factor

sys.path.append('..')
from model import FeatureExtractor, RegionEncoder

sys.path.append('../segment_anything/')
from sam2.build_sam import build_sam2_video_predictor  # type: ignore


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class REN(nn.Module):
    def __init__(self, config):
        super(REN, self).__init__()
        self.exp_dir = os.path.join(config['logging']['save_dir'], config['logging']['exp_name'])
        os.makedirs(self.exp_dir, exist_ok=True)
        print(f'REN Configs: {config}')
        
        # Create the models
        self.extractor_name = config['pretrained']['feature_extractors'][0]
        self.feature_extractor = FeatureExtractor(config, device=device)
        self.region_encoder = RegionEncoder(config).to(device).eval()

        # Load the checkpoint
        self.checkpoint_path = os.path.join(self.exp_dir, 'checkpoint.pth')
        self.load_checkpoint()

        # Image preprocessing transforms
        self.image_resolution = config['parameters']['image_resolution']
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((self.image_resolution, self.image_resolution), antialias=True),
        ])
        self.upsample_features = config['parameters']['upsample_features']
        self.patch_size = config['pretrained']['patch_sizes'][0]
        
        # Grid points for region tokens
        grid_size = config['architecture']['grid_size']
        x_coords = np.linspace(1, self.image_resolution - 2, grid_size, dtype=int)
        y_coords = np.linspace(1, self.image_resolution - 2, grid_size, dtype=int)
        self.grid_points = torch.tensor([(y, x) for y in y_coords for x in x_coords])
    
    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            self.start_epoch = checkpoint['epoch']
            self.start_iter = checkpoint['iter_count']
            self.region_encoder.load_state_dict(checkpoint['region_encoder_state'])
            print(f'Checkpoint loaded from epoch {self.start_epoch}, iteration {self.start_iter}.', self.exp_dir)
        else:
            print('No checkpoint found; exiting..', self.exp_dir)
            exit()

    def translate_point(self, point, original_size, resized_size):
        scale_y = original_size[0] / resized_size[0]
        scale_x = original_size[1] / resized_size[1]
        return point[0] * scale_y, point[1] * scale_x

    def forward(self, images, matching_tokens=None, matching_threshold=0.0, batch_size=4, key_prefix='frame', cache_attn_maps=False):
        image_tokens = {}
        image_height, image_width = images.shape[1], images.shape[2]
        transformed_images = torch.tensor(np.array([self.transform(image) for image in images]))
        image_number = 0
        with torch.no_grad():
            for i in tqdm(range(0, len(transformed_images), batch_size), desc='Processing image frames'):
                image_batch = transformed_images[i:(i + batch_size)].to(device)
                _, feature_maps = self.feature_extractor(self.extractor_name, image_batch, resize=self.upsample_features)

                outputs = self.region_encoder(feature_maps, [self.grid_points for _ in range(image_batch.shape[0])])
                region_tokens = outputs['pred_tokens'].cpu()
                attn_maps = torch.max(F.softmax(outputs['attn_scores'][-1], dim=-1), dim=1)[0].cpu()

                for j in range(image_batch.shape[0]):
                    image_tokens[f'{key_prefix}-{image_number}'] = []
                    image_region_tokens = region_tokens[j]
                    image_attn_maps = attn_maps[j]
                    image_grid_points = self.grid_points

                    if matching_tokens is not None:
                        x = F.normalize(image_region_tokens, p=2, dim=1)
                        y = F.normalize(matching_tokens, p=2, dim=1)
                        matching_scores = torch.mm(x, y.T)
                        matching_scores, _ = torch.max(matching_scores, dim=1)
                        if (matching_scores < matching_threshold).all():
                            best_match_idx = torch.argmax(matching_scores)
                            image_region_tokens = image_region_tokens[best_match_idx:best_match_idx + 1]
                            image_attn_maps = image_attn_maps[best_match_idx:best_match_idx + 1]
                            image_grid_points = image_grid_points[best_match_idx:best_match_idx + 1]
                        else:
                            image_region_tokens = image_region_tokens[matching_scores > matching_threshold]
                            image_attn_maps = image_attn_maps[matching_scores > matching_threshold]
                            image_grid_points = self.grid_points[matching_scores > matching_threshold]

                    for token, map, point in zip(image_region_tokens, image_attn_maps, image_grid_points):
                        y, x = self.translate_point(point, (image_height, image_width), transformed_images.shape[2:4])
                        if cache_attn_maps and self.upsample_features:
                            map_resolution = self.image_resolution
                            attn_map = F.interpolate(map.view(map_resolution, map_resolution)[None][None],
                                                     size=(image_height, image_width), mode='bilinear', align_corners=False)[0][0]
                            image_tokens[f'{key_prefix}-{image_number}'].append({
                                'point': [y, x],
                                'region_feature': token.numpy(),
                                'attn_map': attn_map.numpy(),
                            })
                        elif cache_attn_maps:
                            map_resolution = self.image_resolution // self.patch_size
                            attn_map = F.interpolate(map.view(map_resolution, map_resolution)[None][None],
                                                     size=(image_height, image_width), mode='bilinear', align_corners=False)[0][0]
                            image_tokens[f'{key_prefix}-{image_number}'].append({
                                'point': [y, x],
                                'region_feature': token.numpy(),
                                'attn_map': attn_map.numpy(),
                            })
                        else:
                            image_tokens[f'{key_prefix}-{image_number}'].append({
                                'point': [y, x],
                                'region_feature': token.numpy(),
                            })

                    image_number += 1
                torch.cuda.empty_cache()
        return image_tokens


class QueryEncoder(nn.Module):
    def __init__(self, config):
        super(QueryEncoder, self).__init__()
        self.region_encoder = REN(config['ren'])
        self.cropping_factors = config['visual_query']['query_cropping_factors']

        # Define SAM2 parameters for segmentation mask generation
        self.tracker_param = config['visual_query']['tracker_param']
        self.sam2_ckpt = config['data']['sam2_ckpt']

    def forward(self, record_file, annotation):
        with open(record_file, 'rb') as f:
            record = pickle.load(f)

        # Get query object info
        frames = record['frames']
        visual_crop = annotation['visual_crop']
        query_frame_number = visual_crop['frame_number']
        x = int((visual_crop['x'] / visual_crop['original_width']) * frames.shape[2])
        y = int((visual_crop['y'] / visual_crop['original_height']) * frames.shape[1])
        height = int((visual_crop['height'] / visual_crop['original_height']) * frames.shape[1])
        width = int((visual_crop['width'] / visual_crop['original_width']) * frames.shape[2])
        query_bbox = [x, y, width, height]

        # Get crops for query frames
        query_tokens, query_bboxes, query_frame_numbers, query_frames = [], [], [], []
        query_frame = frames[query_frame_number]
        cropped_query_frames, cropped_query_bboxes = [], []
        for cropping_factor in self.cropping_factors:
            cropped_frame, updated_bbox, _ = crop_using_bbox(query_frame, query_bbox, cropping_factor)
            cropped_query_frames.append(cropped_frame)
            cropped_query_bboxes.append(updated_bbox)
        cropped_query_frames = np.array(cropped_query_frames)
        cropped_query_bboxes = np.array(cropped_query_bboxes)

        # Generate query tokens
        query_tokens = generate_token_from_bbox(cropped_query_frames, cropped_query_bboxes, self.region_encoder, self.sam2_ckpt,
                                                self.tracker_param)
        query_bboxes = torch.cat([torch.tensor(query_bbox)[None]] * len(query_tokens))
        query_frame_numbers = torch.cat([torch.tensor(query_frame_number)[None]] * len(query_tokens))
        query_frames = torch.cat([torch.tensor(query_frame)[None]] * len(query_tokens))
        query_crop = query_frame[y:y + height, x:x + width, :]
        torch.cuda.empty_cache()
        return {
            'query_tokens': query_tokens,
            'query_bboxes': query_bboxes,
            'query_frame_numbers': query_frame_numbers,
            'query_crops': query_crop,
            'query_frames': query_frames,
        }
    

class VideoEncoder(nn.Module):
    def __init__(self, config):
        super(VideoEncoder, self).__init__()
        self.region_encoder = REN(config['ren'])
        self.cropping_factors = config['visual_query']['frame_cropping_factors']

    def forward(self, record_file, annotation, query_tokens=None, selection_top_p=0.0):
        with open(record_file, 'rb') as f:
            record = pickle.load(f)
        query_frame = annotation['query_frame']
        frames = record['frames']
        
        # Create region tokens from cropped frames of the video for better coverage and representation
        video_tokens = {}
        for frame_number in range(query_frame):
            video_tokens[f'frame-{frame_number}'] = []
        for cropping_factor in self.cropping_factors:
            # Get the crops
            crops, crop_starts = [], []
            for frame_number in range(query_frame):
                frame = frames[frame_number]
                crop_size = (int(frame.shape[0] / cropping_factor), int(frame.shape[1] / cropping_factor))
                frame_crops, frame_crop_starts = sliding_window_cropping(frame, crop_size, overlap=0.0)
                crops += frame_crops
                crop_starts += frame_crop_starts
                num_crops_per_frame = len(frame_crops)

            # Get crop tokens
            crop_tokens = self.region_encoder(np.array(crops), query_tokens, selection_top_p, key_prefix='crop')
            for crop_id, crop_start in zip(crop_tokens, crop_starts):
                for object_idx in range(len(crop_tokens[crop_id])):
                    crop_tokens[crop_id][object_idx]['point'][0] = crop_tokens[crop_id][object_idx]['point'][0] / cropping_factor \
                        + crop_start[0]
                    crop_tokens[crop_id][object_idx]['point'][1] = crop_tokens[crop_id][object_idx]['point'][1] / cropping_factor \
                        + crop_start[1]

            # Combine crop tokens with video tokens
            frame_number = 0
            for i in range(0, len(crop_tokens), num_crops_per_frame):
                frame_crop_keys = list(crop_tokens.keys())[i:i + num_crops_per_frame]
                for crop_key in frame_crop_keys:
                    video_tokens[f'frame-{frame_number}'] += crop_tokens[crop_key]
                frame_number += 1

        # Prepare object tokens
        object_tokens, frame_ids, object_points = [], [], []
        for frame_id in video_tokens:
            frame_number = int(frame_id.split('-')[-1])
            frame_tokens = video_tokens[frame_id]
            for object_info in frame_tokens:
                object_tokens.append(object_info['region_feature'])
                object_points.append(object_info['point'])
                frame_ids.append([frame_number])
        
        return {
            'object_tokens': torch.tensor(np.array(object_tokens)),
            'object_points': torch.tensor(np.array(object_points)),
            'frame_ids': torch.tensor(np.array(frame_ids)).flatten(),
            'frames': torch.tensor(frames),
        }


class CandidateSelector(nn.Module):
    def __init__(self):
        super(CandidateSelector, self).__init__()

    def intra_frame_nms(self, object_scores, object_idxs, frame_ids, query_frame_number):
        selected_object_scores = object_scores.clone()
        unique_frame_ids = torch.unique(frame_ids)
        selected_object_idxs = []
        for frame_id in unique_frame_ids:
            idxs = torch.nonzero((frame_ids == frame_id)).flatten()
            frame_object_scores = selected_object_scores[idxs]
            frame_object_idxs = object_idxs[idxs]

            score_threshold = 1.0 * frame_object_scores.max()
            selected_frame_idxs = []
            for idx in range(len(frame_object_scores)):
                if frame_object_scores[idx] >= score_threshold:
                    selected_frame_idxs.append(idx)
            selected_object_idxs.extend(frame_object_idxs[selected_frame_idxs])
        
        selected_object_idxs = torch.tensor(selected_object_idxs)
        selected_object_scores = selected_object_scores[selected_object_idxs]

        # Suppress the query frame detections
        if query_frame_number in unique_frame_ids:
            idx = query_frame_number.item() - unique_frame_ids[0]
            while idx >= 0 and selected_object_scores[idx] >= 0.8:
                selected_object_scores[idx] = 0.0
                idx -= 1
        return selected_object_scores, selected_object_idxs
    
    def inter_frame_nms(self, object_scores, object_idxs, frame_ids, nms_threshold=None, nms_window=None):
        selected_object_scores, selected_object_idxs = [], []

        if nms_threshold is not None:
            while True:
                max_score, max_score_idx = torch.max(object_scores, dim=0)
                if max_score == 0:
                    break
                
                selected_object_scores.append(object_scores[max_score_idx].clone().item())
                selected_object_idxs.append(object_idxs[max_score_idx].clone().item())
                nms_value = nms_threshold * max_score

                # Suppress scores in the previous frames
                idx = max_score_idx - 1
                while idx >= 0 and object_scores[idx] > nms_value:
                    object_scores[idx] = 0
                    idx -= 1

                # Suppress scores in the subsequent frames
                idx = max_score_idx + 1
                while idx < len(object_scores) and object_scores[idx] > nms_value:
                    object_scores[idx] = 0
                    idx += 1
                
                # Suppress the current max score for next iterations
                object_scores[max_score_idx] = 0
            
            selected_object_scores = torch.tensor(selected_object_scores)
            selected_object_idxs = torch.tensor(selected_object_idxs)
            selected_object_idxs, sorted_idxs = torch.sort(selected_object_idxs)
            selected_object_scores = selected_object_scores[sorted_idxs]
        
        elif nms_window is not None:
            selected_object_scores, selected_object_idxs = [], []
            for i in range(0, torch.max(frame_ids) + 1, nms_window):
                idxs = torch.nonzero((frame_ids >= i) & (frame_ids < i + nms_window)).squeeze(dim=1)
                if idxs.numel() == 0:
                    continue
                window_object_scores = object_scores[idxs]
                window_object_idxs = object_idxs[idxs]

                score_threshold = 1.0 * window_object_scores.max()
                selected_window_idxs = []
                for idx in range(len(window_object_scores)):
                    if window_object_scores[idx] >= score_threshold:
                        selected_window_idxs.append(idx)

                selected_object_scores.extend(window_object_scores[selected_window_idxs])
                selected_object_idxs.extend(window_object_idxs[selected_window_idxs])

            selected_object_scores = torch.tensor(selected_object_scores)
            selected_object_idxs = torch.tensor(selected_object_idxs)

        else:
            ValueError('Either nms_threshold or nms_window needs to be specified for inter-frame NMS.')
        return selected_object_scores, selected_object_idxs

    def topk_selection(self, object_scores, object_idxs, top_k):
        if top_k >= len(object_scores):
            return object_scores, object_idxs
        
        topk_threshold = torch.sort(object_scores, descending=True)[0][top_k - 1]
        selected_object_scores, selected_object_idxs = [], []
        for idx in range(len(object_scores)):
            if object_scores[idx] >= topk_threshold:
                selected_object_scores.append(object_scores[idx])
                selected_object_idxs.append(object_idxs[idx])
        
        selected_object_scores = torch.tensor(selected_object_scores)
        selected_object_idxs = torch.tensor(selected_object_idxs)
        return selected_object_scores, selected_object_idxs
    
    def topp_selection(self, object_scores, object_idxs, top_p):
        topp_threshold = top_p 
        if topp_threshold > object_scores.max():
            topp_threshold = object_scores.max()

        selected_object_scores, selected_object_idxs = [], []
        for idx in range(len(object_scores)):
            if object_scores[idx] >= topp_threshold:
                selected_object_scores.append(object_scores[idx])
                selected_object_idxs.append(object_idxs[idx])
        selected_object_scores = torch.tensor(selected_object_scores)
        selected_object_idxs = torch.tensor(selected_object_idxs)
        return selected_object_scores, selected_object_idxs

    def forward(self, object_tokens, query_tokens, object_attn_mask, frame_ids, query_frame_number,
                top_k, top_p, nms_threshold=None, nms_window=None):
        assert object_tokens.shape[0] == 1, 'Only batch size of 1 is supported at the moment.'

        object_tokens = object_tokens[0]
        query_tokens = query_tokens[0]
        object_attn_mask = object_attn_mask[0]
        frame_ids = frame_ids[0]

        x = F.normalize(object_tokens, p=2, dim=1)
        y = F.normalize(query_tokens, p=2, dim=1)
        cosine_scores = torch.mm(x, y.T)
        cosine_scores, query_match_idx = torch.max(cosine_scores, dim=1)
        object_scores = torch.where(object_attn_mask == 1, cosine_scores, 0.0)
        
        # Apply intra-frame NMS to select one object with max score from each frame
        object_idxs = torch.arange(0, len(object_scores))
        selected_object_scores, selected_object_idxs = self.intra_frame_nms(object_scores, object_idxs,
                                                                            frame_ids, query_frame_number)

        # Apply inter-frame NMS to keep only one detection in a sequence of consecutive frames
        if nms_threshold > 0 or nms_window > 1:
            selected_frame_ids = frame_ids[selected_object_idxs]
            selected_object_scores, selected_object_idxs = self.inter_frame_nms(selected_object_scores,
                                                                                selected_object_idxs,
                                                                                selected_frame_ids,
                                                                                nms_threshold=nms_threshold,
                                                                                nms_window=nms_window)

        # Select the top-k objects with max score
        selected_object_scores, selected_object_idxs = self.topk_selection(selected_object_scores,
                                                                           selected_object_idxs, top_k)
        
        # Select the top-p fraction of objects with max score
        selected_object_scores, selected_object_idxs = self.topp_selection(selected_object_scores,
                                                                           selected_object_idxs, top_p)
        
        return {
            'object_scores': object_scores[None],
            'selected_object_scores': selected_object_scores[None],
            'selected_object_idxs': selected_object_idxs[None],
            'query_match_idxs': query_match_idx[selected_object_idxs][None],
        }


class CandidateRefiner(nn.Module):
    def __init__(self, config):
        super(CandidateRefiner, self).__init__()
        self.config = config
        self.cropping_margin_expansion = 1.5
        self.patch_size = config['visual_query']['patch_size']
        self.query_cropping_factors = config['visual_query']['query_cropping_factors']

        # Define region encoder for region token generation
        self.region_encoder = REN(config['ren'])

        # Define SAM2 parameters for segmentation mask generation
        self.tracker_param = config['visual_query']['tracker_param']
        self.sam2_ckpt = config['data']['sam2_ckpt']

    def topp_selection(self, object_scores, object_idxs, top_p):
        topp_threshold = top_p
        if topp_threshold > object_scores.max():
            topp_threshold = object_scores.max()
        
        selected_object_scores, selected_object_idxs = [], []
        for idx in range(len(object_scores)):
            if object_scores[idx] >= topp_threshold:
                selected_object_scores.append(object_scores[idx])
                selected_object_idxs.append(object_idxs[idx])
        selected_object_scores = torch.tensor(selected_object_scores)
        selected_object_idxs = torch.tensor(selected_object_idxs)
        return selected_object_scores, selected_object_idxs

    def forward(self, selected_candidates, top_p=None):
        refined_candidates = []

        # Refine the scores of the candidates selected by the candidate selector
        for candidate in selected_candidates:
            frames = candidate['frames'][0]
            selected_frame_ids = candidate['selected_frame_ids'][0]
            selected_object_points = candidate['selected_object_points'][0]
            selected_object_scores = candidate['selected_object_scores'][0]
            query_tokens = candidate['query_tokens'][0]
            query_match_idxs = candidate['query_match_idxs'][0]
            query_frame_numbers = candidate['query_frame_numbers'][0]
            query_bboxes = candidate['query_bboxes'][0]
            query_timestep = candidate['query_timestep'][0]

            # Crop the query frames and get sam pooled query tokens
            query_frames, cropped_query_frames, updated_query_bboxes = [], [], []
            for query_frame_number, query_bbox in zip(query_frame_numbers, query_bboxes):
                cropping_factor = get_cropping_factor(query_bbox, (frames.shape[2], frames.shape[1]),
                                                      self.cropping_margin_expansion)
                cropped_frame, updated_bbox, _ = crop_using_bbox(frames[query_frame_number], query_bbox, cropping_factor)
                query_frames.append(frames[query_frame_number][None])
                cropped_query_frames.append(cropped_frame[None])
                updated_query_bboxes.append(updated_bbox)
            query_frames = np.concatenate(query_frames)
            cropped_query_frames = np.concatenate(cropped_query_frames)
            sam_pooled_query_tokens = get_sam_pooled_tokens(query_frames, query_bboxes, self.sam2_ckpt, self.tracker_param,
                                                            self.patch_size, self.config)
            sam_pooled_cropped_query_tokens = get_sam_pooled_tokens(cropped_query_frames, updated_query_bboxes, self.sam2_ckpt,
                                                                    self.tracker_param, self.patch_size, self.config)

            # Get object bboxes
            selected_object_bboxes = []
            for frame_id, object_point in zip(selected_frame_ids, selected_object_points):
                object_bbox = point_to_bbox(frames[frame_id].numpy(), object_point, sam_pooled_query_tokens, None, None,
                                            self.sam2_ckpt, self.tracker_param, self.patch_size, self.config)
                selected_object_bboxes.append(object_bbox)

            # Get cropped frames, updated bboxes, and cropping info for the selected video frames
            cropped_frames, updated_bboxes, cropping_infos, crop_frame_ids = [], [], [], []
            for frame_id, object_bbox in zip(selected_frame_ids, selected_object_bboxes):
                for cropping_factor in self.query_cropping_factors:
                    cropped_frame, updated_bbox, cropping_info = crop_using_bbox(frames[frame_id], object_bbox, cropping_factor)
                    cropped_frames.append(cropped_frame[None].numpy())
                    updated_bboxes.append(updated_bbox)
                    cropping_infos.append(cropping_info)
                    crop_frame_ids.append(frame_id)
            cropped_frames = np.concatenate(cropped_frames)
            
            # Get refined candidate tokens from the cropped frames
            crop_tokens = self.region_encoder(cropped_frames, key_prefix='crop')
            num_crops_per_frame = len(self.query_cropping_factors)

            # Re-calculate scores for candidate objects and get refined bboxes
            refined_bboxes, refined_scores = [], []
            for i in range(0, len(crop_tokens), num_crops_per_frame):
                frame_crop_keys = list(crop_tokens.keys())[i:i + num_crops_per_frame]
                relevant_cropping_infos = cropping_infos[i:i + num_crops_per_frame]
                relevant_cropped_frames = cropped_frames[i:i + num_crops_per_frame]

                # Get max scoring candidate from the crops of the current frame
                crop_candidate_tokens, crop_candidate_points, crop_candidate_scores = [], [], []
                for crop_key in frame_crop_keys:
                    # Get all the tokens and points for the current frame crop
                    cropped_frame_tokens, cropped_frame_points = [], []
                    for candidate_info in crop_tokens[crop_key]:
                        cropped_frame_tokens.append(torch.tensor(candidate_info['region_feature'])[None])
                        cropped_frame_points.append(candidate_info['point'])
                    cropped_frame_tokens = torch.cat(cropped_frame_tokens)

                    # Calculate object scores for candidates in the current crop
                    x = F.normalize(cropped_frame_tokens, p=2, dim=-1).to(device)
                    y = F.normalize(query_tokens, p=2, dim=-1).to(device)
                    object_scores = torch.mm(x, y.T)
                    object_scores = torch.max(object_scores, dim=1)[0].squeeze()

                    # Select the object with max score as the new a potential candidate point from current crop
                    max_score_idx = object_scores.argmax()
                    crop_candidate_tokens.append(cropped_frame_tokens[max_score_idx][None])
                    crop_candidate_points.append(cropped_frame_points[max_score_idx])
                    crop_candidate_scores.append(object_scores[max_score_idx])
                crop_candidate_tokens = torch.cat(crop_candidate_tokens)
                crop_candidate_scores = torch.tensor(crop_candidate_scores)

                # Select the new frame candidate and get bbox
                max_score_idx = crop_candidate_scores.argmax()
                max_score_point = crop_candidate_points[max_score_idx]
                max_score_cropped_frame = relevant_cropped_frames[max_score_idx]
                max_score_bbox = point_to_bbox(max_score_cropped_frame, max_score_point, sam_pooled_cropped_query_tokens,
                                               None, None, self.sam2_ckpt, self.tracker_param, self.patch_size, self.config)
                refined_scores.append(crop_candidate_scores.max().item())

                # Undo the cropping transformation for the max score bbox
                cropping_info = relevant_cropping_infos[max_score_idx]
                x = max_score_bbox[0] / cropping_info['scale_x'] + cropping_info['crop_left']
                y = max_score_bbox[1] / cropping_info['scale_y'] + cropping_info['crop_top']
                width = max_score_bbox[2] / cropping_info['scale_x']
                height = max_score_bbox[3] / cropping_info['scale_y']
                refined_bboxes.append(torch.tensor([x, y, width, height])[None])
            refined_bboxes = torch.cat(refined_bboxes)
            refined_scores = torch.tensor(refined_scores)

            refined_candidates.append({
                'frames': frames[None],
                'selected_frame_ids': selected_frame_ids[None],
                'selected_object_points': selected_object_points[None],
                'selected_object_scores': selected_object_scores[None],
                'refined_object_bboxes': refined_bboxes[None],
                'refined_object_scores': refined_scores[None],
                'query_match_idxs': query_match_idxs[None],
                'query_frame_numbers': query_frame_numbers[None],
                'query_bboxes': query_bboxes[None],
                'query_timestep': query_timestep[None],
                'query_tokens': query_tokens[None],
                'response_track': candidate['response_track'],
            })
        
        # Filter out the candidates with low scores
        if top_p is not None:
            for candidate in refined_candidates:
                selected_frame_ids = candidate['selected_frame_ids'][0]
                refined_object_scores = candidate['refined_object_scores'][0]
                refined_object_idxs = torch.arange(0, len(refined_object_scores))

                _, topp_object_idxs = self.topp_selection(refined_object_scores, refined_object_idxs, top_p)
                topp_object_idxs = topp_object_idxs.tolist()
                candidate['selected_frame_ids'] = candidate['selected_frame_ids'][0][topp_object_idxs][None]
                candidate['selected_object_points'] = candidate['selected_object_points'][0][topp_object_idxs][None]
                candidate['selected_object_scores'] = candidate['selected_object_scores'][0][topp_object_idxs][None]
                candidate['refined_object_bboxes'] = candidate['refined_object_bboxes'][0][topp_object_idxs][None]
                candidate['refined_object_scores'] = candidate['refined_object_scores'][0][topp_object_idxs][None]
                candidate['query_match_idxs'] = candidate['query_match_idxs'][0][topp_object_idxs][None]

        return refined_candidates


class VisualQueryTracker(nn.Module):
    def __init__(self, config):
        super(VisualQueryTracker, self).__init__()
        self.config = config

        # Setup SAM2 for tracking
        self.tracker_name = config['visual_query']['tracker_name']
        self.tracker_param = config['visual_query']['tracker_param']
        self.sam2_ckpt = config['data']['sam2_ckpt']
        self.cropping_factors = config['visual_query']['query_cropping_factors']

        # Setup region encoder for generating region tokens
        self.region_encoder = REN(config['ren'])
    
    def forward_tracking(self, frames, selected_frame_ids, selected_object_bboxes):
        transform = T.Compose([T.ToTensor(),
                               T.Resize((1024, 1024), antialias=True),
                               lambda x: x.unsqueeze(0),
                               T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        transformed_frames = []
        for frame in frames:
            transformed_frames.append(transform(frame))
        transformed_frames = torch.cat(transformed_frames, dim=0)
        frame_height, frame_width = frames.shape[1], frames.shape[2]

        # Get tracks for all the selected candidate objects
        predicted_tracks = []
        for selected_frame_id, selected_object_bbox in zip(selected_frame_ids, selected_object_bboxes):
            # Initialize the SAM 2 tracker
            tracker = build_sam2_video_predictor(f'{self.tracker_param}', self.sam2_ckpt, device=device)

            # Run inference
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                half_span_len = 500
                init_start_frame = max(0, selected_frame_id.item() - half_span_len)
                init_end_frame = min(frames.shape[0], selected_frame_id.item() + half_span_len)
                init_frames = transformed_frames[init_start_frame:init_end_frame].to(device)
                inference_state = tracker.init_state(images=init_frames, video_height=frame_height,
                                                     video_width=frame_width)
                tracker.reset_state(inference_state)

                # Add information about the point to track
                x, y, width, height = selected_object_bbox
                _ = tracker.add_new_points_or_box(inference_state=inference_state,
                                                  frame_idx=selected_frame_id - init_start_frame,
                                                  obj_id=0, box=[x, y, x + width, y + height])

                # Track the candidate object in the video frames
                tracked_masks = []
                for frame_id, _, mask_logits in tracker.propagate_in_video(inference_state):
                    object_mask = (mask_logits[0] > 0.0)[0].cpu().numpy()
                    if np.sum(object_mask) == 0:
                        break
                    tracked_masks.append([frame_id + init_start_frame, object_mask])
                predicted_tracks.append(tracked_masks)
                torch.cuda.empty_cache()
        return predicted_tracks
    
    def get_tracked_tokens(self, frames, tracks):
        tracked_tokens = []
        for track in tracks:
            cropped_frames, updated_bboxes = [], []
            for frame_id, object_mask in track:
                object_bbox = mask_to_bbox(object_mask)
                for cropping_factor in self.cropping_factors:
                    cropped_frame, updated_bbox, _ = crop_using_bbox(frames[frame_id], object_bbox, cropping_factor)
                    cropped_frames.append(cropped_frame)
                    updated_bboxes.append(updated_bbox)
            track_tokens = generate_token_from_bbox(np.array(cropped_frames), np.array(updated_bboxes), self.region_encoder,
                                                    self.sam2_ckpt, self.tracker_param)
            tracked_tokens.append(track_tokens)
        return tracked_tokens
    
    def track(self, frames, selected_frame_ids, selected_object_bboxes, selected_object_scores, query_timestep,
              get_tokens=True, top_p=0.75):
        if self.tracker_name == 'sam2':
            # Get tracks in the subsequent frames
            future_tracks = self.forward_tracking(frames[:(query_timestep + 1)], selected_frame_ids, 
                                                  selected_object_bboxes)

            # Get tracks in the preceeding frames
            reversed_frames = frames[:(query_timestep + 1)][::-1]
            reversed_frame_ids = query_timestep - selected_frame_ids
            past_tracks = self.forward_tracking(reversed_frames, reversed_frame_ids, selected_object_bboxes)
            for track_idx in range(len(past_tracks)):
                for object_idx in range(len(past_tracks[track_idx])):
                    past_tracks[track_idx][object_idx][0] = query_timestep - past_tracks[track_idx][object_idx][0]

            # Combine past and future tracks
            tracked_masks = []
            for i in range(len(selected_frame_ids)):
                tracked_masks.append(past_tracks[i][::-1][:-1] + future_tracks[i])

            # Get region tokens for the tracked object
            if get_tokens:
                tracked_tokens = self.get_tracked_tokens(frames, tracked_masks)
            else:
                tracked_tokens = [None] * len(tracked_masks)
            track_scores = [selected_object_scores.item() for _ in range(len(tracked_masks))]
                
            # Select one track as the final prediction
            max_score = max(track_scores)
            score_threshold = top_p * max_score
            for idx in range(len(track_scores) - 1, -1, -1):
                if track_scores[idx] >= score_threshold:
                    predicted_track = tracked_masks[idx]
                    predicted_track_score = track_scores[idx]
                    predicted_track_tokens = tracked_tokens[idx]
                    break

            # Convert the track from masks to bboxes
            for track_idx in range(len(predicted_track)):
                frame_id, object_mask = predicted_track[track_idx]
                object_bbox = mask_to_bbox(object_mask).tolist()
                predicted_track[track_idx] = [frame_id] + object_bbox

            return {
                'predicted_track': predicted_track,
                'predicted_track_score': predicted_track_score,
                'predicted_track_tokens': predicted_track_tokens,
            }

        else:
            raise NotImplementedError(f'{self.tracker_name} not implemented.')