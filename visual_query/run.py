import os
import sys
import yaml
import time
import json
import numpy as np
import random
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import patches
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader import VisualQueryDataset
from models import CandidateSelector, CandidateRefiner, VisualQueryTracker
from metrics import BBox, ResponseTrack, compute_visual_query_metrics
from vq_utils import is_blurry, print_log, format_predictions


device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class CandidateSelection():
    def visualize_recalled_frames(self, object_scores, frame_ids, object_points, frames, query_frame_numbers,
                                  query_bboxes, save_dir='.'):
        batch_size, num_objects = object_scores.shape
        num_queries = query_frame_numbers.shape[1]
        os.makedirs(save_dir, exist_ok=True)

        for batch_idx in range(batch_size):
            object_scores_batch = object_scores[batch_idx]
            frame_ids_batch = frame_ids[batch_idx]
            object_points_batch = object_points[batch_idx]
            frames_batch = frames[batch_idx]
            query_frame_numbers_batch = query_frame_numbers[batch_idx]
            query_bboxes_batch = query_bboxes[batch_idx]

            # Visualize the video frames with bbox on the selected object
            for object_idx in range(num_objects):
                frame_id = frame_ids_batch[object_idx]
                object_score = object_scores_batch[object_idx]
                object_point = object_points_batch[object_idx].cpu()
                frame = frames_batch[frame_id]

                plt.imshow(frame.numpy())
                plt.axis('off')
                plt.scatter([object_point[1]], [object_point[0]], marker='o', s=25, c='cyan')
                plt.title(f'{object_score:.4f}')
                plt.savefig(os.path.join(save_dir, f'{frame_id}.jpg'))
                plt.clf()
            
            # Visualize the query object
            for query_idx in range(num_queries):
                query_frame_id = query_frame_numbers_batch[query_idx].item()
                plt.imshow(frames_batch[query_frame_id])
                plt.axis('off')
                x, y, width, height = query_bboxes_batch[query_idx]
                rectangle = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='lime', facecolor='none')
                plt.gca().add_patch(rectangle)
                plt.savefig(os.path.join(save_dir, f'query-{query_frame_id}.jpg'))
                plt.clf()
    
    def run(self, config, object_tokens, query_tokens, object_attn_mask, frame_ids, query_frame_numbers, object_points,
            frames, query_bboxes, query_timestep, response_track, visualize_results=False, save_dir='.'):
        model = CandidateSelector()
        selected_object_info = model(object_tokens, query_tokens, object_attn_mask, frame_ids, query_frame_numbers[0][0],
                                     top_k=config['selection_top_k'], top_p=config['selection_top_p'],
                                     nms_threshold=config['nms_threshold'])
        selected_object_scores = selected_object_info['selected_object_scores']
        selected_object_idxs = selected_object_info['selected_object_idxs']
        query_match_idxs = selected_object_info['query_match_idxs']
        
        selected_object_points = torch.gather(object_points, 1, selected_object_idxs.unsqueeze(2).expand(-1, -1, 2))
        selected_frame_ids = torch.gather(frame_ids, 1, selected_object_idxs)
        selected_candidates = {
            'frames': frames,
            'selected_object_points': selected_object_points,
            'selected_object_scores': selected_object_scores,
            'selected_object_idxs': selected_object_idxs,
            'selected_frame_ids': selected_frame_ids,
            'query_tokens': query_tokens,
            'query_match_idxs': query_match_idxs,
            'query_frame_numbers': query_frame_numbers,
            'query_bboxes': query_bboxes,
            'query_timestep': query_timestep,
            'response_track': response_track,
        }

        # Visualize the recalled frames and objects
        if visualize_results:
            self.visualize_recalled_frames(selected_object_scores, selected_frame_ids, selected_object_points,
                                           frames, query_frame_numbers, query_bboxes, save_dir=save_dir)
        
        return selected_candidates


class CandidateRefinement():
    def visualize_refined_bboxes(self, frames, selected_frame_ids, selected_object_points, selected_object_scores,
                                 refined_bboxes, refined_scores, query_bboxes, query_frame_numbers, save_dir='.'):
        os.makedirs(save_dir, exist_ok=True)

        # Visualize the original selected bboxes and the refined bboxes
        for i in range(refined_bboxes.shape[0]):
            frame = frames[selected_frame_ids[i]]
            plt.imshow(frame)
            plt.axis('off')
            plt.title(f'{selected_object_scores[i]:.4f} -> {refined_scores[i]:.4f}')

            y, x = selected_object_points[i]
            plt.scatter([x], [y], s=25, c='cyan')

            x, y, width, height = refined_bboxes[i]
            rectangle = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='lime', facecolor='none')
            plt.gca().add_patch(rectangle)
            plt.savefig(f'{save_dir}/{selected_frame_ids[i]}.jpg')
            plt.clf()
        
        # Visualize the query bbox
        for i in range(query_frame_numbers.shape[0]):
            plt.imshow(frames[query_frame_numbers[i]])
            plt.axis('off')
            x, y, width, height = query_bboxes[i]
            rectangle = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='b', facecolor='none')
            plt.gca().add_patch(rectangle)
            plt.savefig(f'{save_dir}/query-{query_frame_numbers[i]}.jpg')
            plt.clf()
    
    def run(self, config, selected_candidates, visualize_results=False, save_dir='.'):
        model = CandidateRefiner(config)
        refined_candidates = model(selected_candidates, config['visual_query']['refinement_top_p'])

        # Visualize the refined candidate objects and bboxes
        if visualize_results:
            for candidate in refined_candidates:
                frames = candidate['frames'][0]
                selected_frame_ids = candidate['selected_frame_ids'][0]
                selected_object_points = candidate['selected_object_points'][0]
                selected_object_scores = candidate['selected_object_scores'][0]
                refined_object_bboxes = candidate['refined_object_bboxes'][0]
                refined_object_scores = candidate['refined_object_scores'][0]
                query_frame_numbers = candidate['query_frame_numbers'][0]
                query_bboxes = candidate['query_bboxes'][0]
                self.visualize_refined_bboxes(frames, selected_frame_ids, selected_object_points, selected_object_scores,
                                              refined_object_bboxes, refined_object_scores, query_bboxes,
                                              query_frame_numbers, save_dir=save_dir)
        
        # Update the selected object info to the refined object info
        for candidate in refined_candidates:
            candidate['selected_object_bboxes'] = candidate['refined_object_bboxes']
            candidate['selected_object_scores'] = candidate['refined_object_scores']
            del candidate['refined_object_bboxes']
            del candidate['refined_object_scores']
        return refined_candidates


class TrackPrediction():
    def visualize_track(self, frames, predicted_track, score, response_track, query_frame_id, query_bbox, save_dir='.'):
        os.makedirs(save_dir, exist_ok=True)
        frame_to_bbox = {}
        for t in predicted_track:
            frame_id = t[0]
            frame_to_bbox[frame_id] = {'prediction': [int(t[1]), int(t[2]), int(t[3]), int(t[4])]}
        for t in response_track:
            frame_id = t[0].item()
            if frame_id in frame_to_bbox:
                frame_to_bbox[frame_id]['target'] = [int(t[1]), int(t[2]), int(t[3]), int(t[4])]
            else:
                frame_to_bbox[frame_id] = {'target': [int(t[1]), int(t[2]), int(t[3]), int(t[4])]} 

        # Visualize the predicted and the ground truth tracks
        for frame_id in frame_to_bbox:
            plt.imshow(frames[frame_id])
            plt.axis('off')

            if 'prediction' in frame_to_bbox[frame_id]:
                x, y, width, height = frame_to_bbox[frame_id]['prediction']
                rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='cyan', facecolor='none')
                plt.gca().add_patch(rect)

            if 'target' in frame_to_bbox[frame_id]:
                x, y, width, height = frame_to_bbox[frame_id]['target']
                rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='lime', facecolor='none')
                plt.gca().add_patch(rect)
            
            plt.title(f'score: {score:.4f}')
            plt.savefig(f'{save_dir}/{frame_id}.jpg')
            plt.clf()
        
        # Visualize the query
        plt.imshow(frames[query_frame_id])
        plt.axis('off')
        x, y, width, height = query_bbox
        rectangle = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='b', facecolor='none')
        plt.gca().add_patch(rectangle)
        plt.savefig(f'{save_dir}/query-{query_frame_id}.jpg')
        plt.clf()

    def combine_tracks(self, tracking_results):
        combined_track, combined_track_tokens, tracked_frames = [], [], []
        combined_track_score = 0.0

        for track_idx in range(len(tracking_results)):
            predicted_track = tracking_results[track_idx]['predicted_track']
            combined_track_score += tracking_results[track_idx]['predicted_track_score']
            predicted_track_tokens = tracking_results[track_idx]['predicted_track_tokens']
            for candidate, token in zip(predicted_track, predicted_track_tokens):
                if candidate[0] not in tracked_frames:
                    tracked_frames.append(candidate[0])
                    combined_track.append(candidate)
                    combined_track_tokens.append(token[None])
        combined_track_score /= len(tracking_results)
        combined_track_tokens = torch.cat(combined_track_tokens)
        return {
            'predicted_track': combined_track,
            'predicted_track_score': combined_track_score,
            'predicted_track_tokens': combined_track_tokens,
        }

    def run(self, config, candidates, to_track='last', get_tokens=True, visualize_final_tracks=False, save_dir='.'):
        tracker = VisualQueryTracker(config)

        frames = candidates['frames'][0].numpy()
        selected_object_bboxes = candidates['selected_object_bboxes'][0]
        selected_frame_ids = candidates['selected_frame_ids'][0]
        selected_object_scores = candidates['selected_object_scores'][0]
        query_frame_id = candidates['query_frame_numbers'][0][0]
        query_bbox = candidates['query_bboxes'][0][0]
        query_timestep = candidates['query_timestep'][0].item()
        response_track = candidates['response_track'][0]
        if 'original_query_frame_numbers' in candidates:
            original_query_frame_id = candidates['original_query_frame_numbers'][0][0]
        else:
            original_query_frame_id = query_frame_id

        # Use tracker to compute candidate tracks
        if to_track == 'last':
            tracking_results = tracker.track(frames, selected_frame_ids[-1:], selected_object_bboxes[-1:], 
                                             selected_object_scores[-1:], query_timestep, get_tokens=get_tokens,
                                             top_p=config['visual_query']['tracking_top_p'])
        elif to_track == 'max':
            idx = torch.argmax(selected_object_scores)
            tracking_results = tracker.track(frames, selected_frame_ids[idx:idx + 1], selected_object_bboxes[idx:idx + 1], 
                                             selected_object_scores[idx:idx + 1], query_timestep, get_tokens=get_tokens, 
                                             top_p=config['visual_query']['tracking_top_p'])
        elif to_track == 'all':
            tracking_results = []
            for i in range(selected_frame_ids.shape[0]):
                tracking_result = tracker.track(frames, selected_frame_ids[i:i + 1], selected_object_bboxes[i:i + 1], 
                                                selected_object_scores[i:(i + 1)], query_timestep, get_tokens=get_tokens,
                                                top_p=config['visual_query']['tracking_top_p'])
                tracking_results.append(tracking_result)
            tracking_results = self.combine_tracks(tracking_results)
        predicted_track = tracking_results['predicted_track']
        predicted_track_score = tracking_results['predicted_track_score']
        tracked_object_tokens = tracking_results['predicted_track_tokens']

        # Remove the original query frame if it is a part of the predicted track
        if to_track == 'last':
            query_frame_idx = None
            for object_idx in range(len(predicted_track)):
                if predicted_track[object_idx][0] == original_query_frame_id:
                    query_frame_idx = object_idx
                    break
            if query_frame_idx is not None:
                if query_frame_idx == len(predicted_track) - 1:
                    predicted_track_score = 0.0
                else:
                    if tracked_object_tokens is not None:
                        num_tokens_per_frame = len(tracked_object_tokens) // len(predicted_track)
                        tracked_object_tokens = tracked_object_tokens[num_tokens_per_frame * (query_frame_idx + 1):]
                    predicted_track = predicted_track[query_frame_idx + 1:]

        # Visualize the predicted and the target tracks
        if visualize_final_tracks == True:
            self.visualize_track(frames, predicted_track, predicted_track_score, response_track, query_frame_id,
                                 query_bbox, save_dir)

        return {
            'predicted_track': predicted_track,
            'predicted_track_score': predicted_track_score,
            'tracked_object_tokens': tracked_object_tokens,
        }


class QueryExpansion():
    def visualize_queries(self, frames, query_frame_numbers, query_bboxes, query_scores, save_dir='.'):
        os.makedirs(save_dir, exist_ok=True)
        for frame_id, bbox, score in zip(query_frame_numbers, query_bboxes, query_scores):
            plt.imshow(frames[frame_id])
            plt.title(f'{score:.4f}')
            plt.axis('off')
            x, y, width, height = bbox
            rectangle = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rectangle)
            plt.savefig(os.path.join(save_dir, f'{frame_id}.jpg'))
            plt.clf()

    def run(self, batch, predicted_track, tracked_object_tokens, query_selection_threshold=0.5, bbox_size_threshold=10,
            visualize_query_pool=False, save_dir='.'):
        last_tracked_frame = predicted_track[-1][0]
        if last_tracked_frame >= batch['frame_ids'][0][-1]:
            return None

        # Update batch to contain only objects that appear after the last tracked frame
        future_frame_tokens_idx = (batch['frame_ids'][0] == last_tracked_frame + 1).nonzero(as_tuple=True)[0][0].item()
        batch['object_tokens'] = batch['object_tokens'][:, future_frame_tokens_idx:]
        batch['object_points'] = batch['object_points'][:, future_frame_tokens_idx:]
        batch['object_attn_mask'] = batch['object_attn_mask'][:, future_frame_tokens_idx:]
        batch['frame_ids'] = batch['frame_ids'][:, future_frame_tokens_idx:]
        num_queries = batch['query_tokens'].shape[1]

        # Calculate the similarity between the original query and the potential queries
        query_tokens = batch['query_tokens'][0]
        x = F.normalize(tracked_object_tokens, p=2, dim=-1).to(device)
        y = F.normalize(query_tokens, p=2, dim=-1).to(device)
        cosine_similarity = torch.mm(x, y.T)
        cosine_similarity = torch.max(cosine_similarity, dim=1)[0]
        selected_query_idxs = torch.where(cosine_similarity > query_selection_threshold)[0].tolist()

        # Get updated pool of query tokens
        updated_query_tokens, updated_query_frame_numbers, updated_query_bboxes = [], [], []
        num_tokens_per_frame = len(tracked_object_tokens) // len(predicted_track)
        for query_idx in range(len(tracked_object_tokens)):
            track_idx = int(query_idx / num_tokens_per_frame)
            candidate = predicted_track[track_idx]
            frame_number = candidate[0]
            candidate_bbox = candidate[1:]
            if min(candidate_bbox[2], candidate_bbox[3]) <= bbox_size_threshold or \
                is_blurry(batch['frames'][0][frame_number].numpy()):
                continue
            if query_idx in selected_query_idxs:
                updated_query_tokens.append(tracked_object_tokens[track_idx][None][None])
                updated_query_frame_numbers.append(candidate[0])
                updated_query_bboxes.append(candidate_bbox)
        if len(updated_query_tokens):
            updated_query_tokens = torch.cat(updated_query_tokens, dim=1)
            updated_query_frame_numbers = torch.tensor(updated_query_frame_numbers)[None]
            updated_query_bboxes = torch.tensor(updated_query_bboxes)[None]
            batch['query_tokens'] = torch.cat([batch['query_tokens'], updated_query_tokens], dim=1)
            batch['query_frame_numbers'] = torch.cat([batch['query_frame_numbers'], updated_query_frame_numbers], dim=1)
            batch['query_bboxes'] = torch.cat([batch['query_bboxes'], updated_query_bboxes], dim=1)

        if visualize_query_pool:
            selected_query_scores = [1] * num_queries + cosine_similarity[selected_query_idxs].tolist()
            self.visualize_queries(batch['frames'][0], batch['query_frame_numbers'][0], batch['query_bboxes'][0],
                                   selected_query_scores, save_dir)
        return batch


class Evaluator():
    def run(self, selected_candidates, predicted_tracks, predicted_track_scores, video_uids, clip_uids, qset_ids):
        # Get ground truth response tracks in the desired format
        ground_truth_response_track, visual_crop_boxes, frame_dimensions = [], [], []
        for item in selected_candidates:
            frames = item['frames']
            frame_height, frame_width = frames.shape[2], frames.shape[3]
            frame_dimensions.append((frame_height, frame_width))

            response_track = item['response_track'][0].tolist()
            rt_bboxes = [BBox(gt[0], gt[1], gt[2], gt[1] + gt[3], gt[2] + gt[4]) for gt in response_track]
            ground_truth_response_track.append(ResponseTrack(rt_bboxes))

            query_bbox = item['query_bboxes'][0][0]
            query_frame_number = item['query_frame_numbers'][0][0]
            qb_bbox = BBox(query_frame_number,
                           query_bbox[0], query_bbox[1], query_bbox[0] + query_bbox[2], query_bbox[1] + query_bbox[3])
            visual_crop_boxes.append(qb_bbox)

        # Get predicted response tracks in the desired format
        predicted_response_track = []
        for rt, score in zip(predicted_tracks, predicted_track_scores):
            bboxes = [BBox(pt[0], pt[1], pt[2], pt[1] + pt[3], pt[2] + pt[4]) for pt in rt]
            predicted_response_track.append([ResponseTrack(bboxes, score)])

        # Compute metrics
        metrics = compute_visual_query_metrics(predicted_response_track, ground_truth_response_track,
                                               visual_crop_boxes)
        predictions = format_predictions(video_uids, clip_uids, qset_ids, predicted_response_track,
                                         ground_truth_response_track, frame_dimensions)
        return metrics, predictions
    

def run(config, refine_selected_candidates=True, refine_tracks=True, cache_predictions=True, visualize_results=False):
    exp_dir = os.path.join(config['visual_query']['save_dir'], config['visual_query']['exp_name'])
    os.makedirs(exp_dir, exist_ok=True)
    predictions_dir = os.path.join(exp_dir, 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)
    print_log(f'Configs: {config}', exp_dir)
    run_id = 0
    if len(sys.argv) == 2:
        run_id = int(sys.argv[1])

    # Create data loader
    if config['visual_query']['split'] == 'train':
        dataset = VisualQueryDataset(data_dir=config['data']['train_data_dir'], 
                                     num_examples=config['data']['num_train_examples'],
                                     run_id=run_id, config=config)
        dataloader = DataLoader(dataset, batch_size=config['visual_query']['batch_size'], num_workers=0)
        
    elif config['visual_query']['split'] == 'val':
        dataset = VisualQueryDataset(data_dir=config['data']['val_data_dir'],
                                     num_examples=config['data']['num_val_examples'],
                                     run_id=run_id, config=config)
        dataloader = DataLoader(dataset, batch_size=config['visual_query']['batch_size'], num_workers=0)
    
    elif config['visual_query']['split'] == 'test':
        dataset = VisualQueryDataset(data_dir=config['data']['test_data_dir'],
                                     num_examples=config['data']['num_test_examples'],
                                     run_id=run_id, config=config)
        dataloader = DataLoader(dataset, batch_size=config['visual_query']['batch_size'], num_workers=0)
        
    # Instantiate the modules
    candidate_selection = CandidateSelection()
    candidate_refinement = CandidateRefinement()
    track_prediction = TrackPrediction()
    query_expansion = QueryExpansion()
    evaluator = Evaluator()

    all_selected_candidates, all_predicted_tracks, all_predicted_track_scores = [], [], []
    all_video_uids, all_clip_uids, all_qset_ids = [], [], []

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        object_tokens = batch['object_tokens']
        object_points = batch['object_points']
        object_attn_mask = batch['object_attn_mask']
        query_tokens = batch['query_tokens']
        query_bboxes = batch['query_bboxes']
        query_frame_numbers = batch['query_frame_numbers']
        query_timestep = batch['query_timestep']
        frame_ids = batch['frame_ids']
        frames = batch['frames']
        response_track = batch['response_track']
        video_uid = batch['video_uid']
        clip_uid = batch['clip_uid']
        qset_id = batch['qset_id']

        # Select candidate objects using candidate selector
        selected_candidates = candidate_selection.run(config['visual_query'], object_tokens, query_tokens,
                                                      object_attn_mask, frame_ids, query_frame_numbers,
                                                      object_points, frames, query_bboxes, query_timestep,
                                                      response_track, visualize_results=visualize_results,
                                                      save_dir=os.path.join(exp_dir, f'{batch_idx}/0-selector'))

        # Refine candidate objects using candidate refiner
        if refine_selected_candidates:
            selected_candidates = candidate_refinement.run(config, [selected_candidates], visualize_results=visualize_results,
                                                           save_dir=os.path.join(exp_dir, f'{batch_idx}/0-refiner'))[0]
        
        # Track the latest candidate using tracker
        track_predictions = track_prediction.run(config, selected_candidates, visualize_final_tracks=visualize_results,
                                                 to_track='max', save_dir=os.path.join(exp_dir, f'{batch_idx}/0-tracker'))
        predicted_track = track_predictions['predicted_track']
        predicted_track_score = track_predictions['predicted_track_score']
        tracked_object_tokens = track_predictions['tracked_object_tokens']

        # Refine tracks using all previously tracked objects as queries
        if refine_tracks:
            updated_batch = query_expansion.run(batch, predicted_track, tracked_object_tokens, 
                                                config['visual_query']['query_selection_threshold'],
                                                visualize_query_pool=visualize_results,
                                                save_dir=os.path.join(exp_dir, f'{batch_idx}/1-queries'))
            if updated_batch is not None:
                object_tokens = updated_batch['object_tokens']
                object_points = updated_batch['object_points']
                object_attn_mask = updated_batch['object_attn_mask']
                query_tokens = updated_batch['query_tokens']
                query_bboxes = updated_batch['query_bboxes']
                query_frame_numbers = updated_batch['query_frame_numbers']
                query_timestep = updated_batch['query_timestep']
                frame_ids = updated_batch['frame_ids']
                frames = updated_batch['frames']
                response_track = updated_batch['response_track']

                # Select candidate objects using candidate selector
                updated_candidates = candidate_selection.run(config['visual_query'], object_tokens, query_tokens,
                                                             object_attn_mask, frame_ids, query_frame_numbers,
                                                             object_points, frames, query_bboxes, query_timestep,
                                                             response_track, visualize_results=visualize_results,
                                                             save_dir=os.path.join(exp_dir, f'{batch_idx}/1-selector'))

                # Refine candidate objects using candidate refiner
                if refine_selected_candidates:
                    updated_candidates = candidate_refinement.run(config, [updated_candidates], visualize_results=visualize_results,
                                                                  save_dir=os.path.join(exp_dir, f'{batch_idx}/1-refiner'))
                    updated_candidates = updated_candidates[0]
                
                # Skip tracking if the candidate score is too low
                selected_object_scores = updated_candidates['selected_object_scores'][0]
                idxs = selected_object_scores >= config['visual_query']['track_selection_threshold']
                if idxs.any():
                    updated_candidates['selected_frame_ids'] = updated_candidates['selected_frame_ids'][:, idxs]
                    updated_candidates['selected_object_bboxes'] = updated_candidates['selected_object_bboxes'][:, idxs]
                    updated_candidates['selected_object_points'] = updated_candidates['selected_object_points'][:, idxs]
                    updated_candidates['selected_object_scores'] = updated_candidates['selected_object_scores'][:, idxs]
                    updated_candidates['query_match_idxs'] = updated_candidates['query_match_idxs'][:, idxs]
                    latest_query_match_idx = updated_candidates['query_match_idxs'][0][-1].item()
                    latest_query_frame_number = updated_candidates['query_frame_numbers'][0][latest_query_match_idx]
                    latest_query_bbox = updated_candidates['query_bboxes'][0][latest_query_match_idx]
                    updated_candidates['query_frame_numbers'] = torch.tensor([[latest_query_frame_number]])
                    updated_candidates['query_bboxes'] = latest_query_bbox[None][None]
                    updated_candidates['original_query_frame_numbers'] = selected_candidates['query_frame_numbers']
                    
                    # Track the latest candidate using tracker
                    updated_track_predictions = track_prediction.run(config, updated_candidates,
                                                                     visualize_final_tracks=visualize_results, get_tokens=False,
                                                                     save_dir=os.path.join(exp_dir, f'{batch_idx}/1-tracker'))
                    updated_track = updated_track_predictions['predicted_track']
                    updated_track_score = updated_track_predictions['predicted_track_score']

                    if updated_track_score > config['visual_query']['track_selection_threshold'] * predicted_track_score:
                        selected_candidates = updated_candidates
                        predicted_track = updated_track
                        predicted_track_score = updated_track_score
        
        # Cache the predictions for evaluation
        all_selected_candidates.append(selected_candidates)
        all_predicted_tracks.append(predicted_track)
        all_predicted_track_scores.append(predicted_track_score)
        all_video_uids.append(video_uid[0])
        all_clip_uids.append(clip_uid[0])
        all_qset_ids.append(qset_id[0])
        
    # Evaluate the predictions
    metrics, _ = evaluator.run(all_selected_candidates, all_predicted_tracks, all_predicted_track_scores,
                               all_video_uids, all_clip_uids, all_qset_ids)
    for area_range, computed_metrics in metrics.items():
        print_log(f'Area range: {area_range}', exp_dir)
        for metric_name, metric_value in computed_metrics.items():
            print_log(f'  {metric_name} \t-\t {metric_value:.4f}', exp_dir)
    
    # Cache the predictions
    if cache_predictions:
        for clip_uid in set(all_clip_uids):
            clip_selected_candidates = [all_selected_candidates[i]
                                        for i in range(len(all_clip_uids)) if all_clip_uids[i] == clip_uid]
            clip_predicted_tracks = [all_predicted_tracks[i]
                                      for i in range(len(all_clip_uids)) if all_clip_uids[i] == clip_uid]
            clip_predicted_track_scores = [all_predicted_track_scores[i]
                                            for i in range(len(all_clip_uids)) if all_clip_uids[i] == clip_uid]
            clip_video_uids = [all_video_uids[i] for i in range(len(all_clip_uids)) if all_clip_uids[i] == clip_uid]
            clip_qset_ids = [all_qset_ids[i] for i in range(len(all_clip_uids)) if all_clip_uids[i] == clip_uid]
            _, clip_predictions = evaluator.run(clip_selected_candidates, clip_predicted_tracks, 
                                                clip_predicted_track_scores, clip_video_uids,
                                                [clip_uid] * len(clip_video_uids), clip_qset_ids)
            with open(os.path.join(predictions_dir, f'{run_id}-{clip_uid}.json'), 'w') as f:
                json.dump(clip_predictions, f)


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    exp_dir = os.path.join(config['visual_query']['save_dir'], config['visual_query']['exp_name'])
    os.makedirs(exp_dir, exist_ok=True)
    start = time.time()
    run(config)
    print_log(f'Run time: {time.time() - start:.4f} seconds', exp_dir)