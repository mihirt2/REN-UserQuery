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
    assert len(layers) == 1, 'Implemented for single layer extraction only.'

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

            B, _, C = features_out.size()
            H, W = image_batch.shape[2], image_batch.shape[3]
            patch_H, patch_W = math.ceil(H / patch_length), math.ceil(W / patch_length)
            features_out = features_out.permute(0, 2, 1).view(B, C, patch_H, patch_W)
            features.append(features_out)
            torch.cuda.empty_cache()
    features = torch.cat(features, dim=0)
    return features.detach().cpu().to(torch.float32)


def extract_dino_v2(model, images, batch_size=128, patch_length=14, layers=[23]):
    transform = T.Compose([T.ToTensor(),
                           lambda x: x.unsqueeze(0),
                           CenterPadding(multiple=patch_length),
                           T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    
    transformed_images = []
    for image in images:
        transformed_images.append(transform(image).to(device=device, dtype=torch.bfloat16))
    transformed_images = torch.cat(transformed_images, dim=0)
    
    features = []
    for i in range(0, transformed_images.shape[0], batch_size):
        image_batch = transformed_images[i:(i + batch_size)]
        with torch.inference_mode():
            features_out = model.get_intermediate_layers(image_batch, n=layers, reshape=True)
            features.append(torch.cat(features_out, dim=1))
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


def extract_image_features(images, config):
    feature_extractor = config['visual_query']['feature_extractor']
    if feature_extractor == 'dino':
        print('Extracting features using DINO')
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
        model = model.to(device=device, dtype=torch.bfloat16)
        features = extract_dino(model, images)

    elif feature_extractor == 'dinov2':
        print('Extracting features using DINOv2')
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        model = model.to(device=device, dtype=torch.bfloat16)
        features = extract_dino_v2(model, images)
    return features


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
    predictor = SAM2ImagePredictor(build_sam2(f'{tracker_param}', sam2_ckpt, device=device))
    for frame, bbox in zip(frames, bboxes):
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            predictor.set_image(cv2.GaussianBlur(frame, (3, 3), 0))
            x, y, width, height = bbox
            mask, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=np.array([x, y, x + width, y + height])[None],
                multimask_output=False,
            )
            masks.append(refine_mask(mask[0]))
    return masks


def get_sam_region_from_points(sam2_ckpt, tracker_param, frames, points):
    masks = []
    predictor = SAM2ImagePredictor(build_sam2(f'{tracker_param}', sam2_ckpt, device=device))
    for frame, point in zip(frames, points):
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            predictor.set_image(cv2.GaussianBlur(frame, (3, 3), 0))
            y, x = point
            mask, _, _ = predictor.predict(
                point_coords=np.array([[x, y]]),
                point_labels=np.array([1]),
                box=None,
                multimask_output=True,
            )
            mask = [refine_mask(m) for m in mask]
            masks.append(mask)
    return masks


def generate_frame_tokens(feature, masks, pooling_method='average'):
    frame_tokens = []
    feature_dim = feature.shape[0]
    for mask in masks:
        sam_mask = mask['segmentation'].cpu()
        r_1, r_2 = np.where(sam_mask == 1)

        if pooling_method == 'average':
            features_in_sam = feature[:, r_1, r_2].view(feature_dim, -1).mean(1).detach()
        elif pooling_method == 'max':
            features_in_sam, _ = torch.max(feature[:, r_1, r_2].view(feature_dim, -1), 1)
        
        frame_tokens.append({
            'region_feature': features_in_sam,
            'mask': sam_mask,
            'bbox': mask_to_bbox(sam_mask),
        })
    return frame_tokens


def generate_token_from_bbox(frames, bboxes, region_encoder, sam2_ckpt, tracker_param, return_attn_maps=False):
    # Get the masks and region tokens for the frame
    frame_tokens = region_encoder(frames)
    selected_masks = get_sam_region_from_bbox(sam2_ckpt, tracker_param, frames, bboxes)

    # Generate the tokens
    tokens, attn_maps = [], []
    for frame_tokens, mask in zip(frame_tokens.values(), selected_masks):
        if np.sum(mask) == 0:
            continue
        selected_token, selected_attn_map = None, None
        dist_transform = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        point_dist = None
        for object_info in frame_tokens:
            object_point = object_info['point']
            point_y, point_x = int(object_point[0]), int(object_point[1])
            if mask[point_y, point_x] and (point_dist == None or dist_transform[point_y, point_x] > point_dist):
                selected_token = torch.tensor(object_info['region_feature'])[None]
                if return_attn_maps:
                    selected_attn_map = object_info['attn_map']
                point_dist = dist_transform[point_y, point_x]
        tokens.append(selected_token)
        if return_attn_maps:
            attn_maps.append(selected_attn_map)
        torch.cuda.empty_cache()

    # Handle cases where no token is generated
    reference_tensor = None
    for token in tokens:
        if token is not None:
            reference_tensor = torch.zeros_like(token)
            break
    for i in range(len(tokens)):
        if tokens[i] is None:
            tokens[i] = reference_tensor
    tokens = torch.cat(tokens)

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
    rows, cols = np.where(mask == 1)
    x_min = np.min(cols)
    y_min = np.min(rows)
    x_max = np.max(cols)
    y_max = np.max(rows)
    width = x_max - x_min
    height = y_max - y_min
    return torch.tensor([x_min, y_min, width, height])


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