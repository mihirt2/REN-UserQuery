import os
import pickle
import torch
from torch.utils.data import Dataset
from models import VideoEncoder, QueryEncoder


class VisualQueryDataset(Dataset):
    def __init__(self, data_dir, num_examples, config, run_id):
        self.data = []
        for f in sorted(os.listdir(data_dir))[(run_id * num_examples):((run_id + 1) * num_examples)]:
            exp_dir = os.path.join(config['visual_query']['save_dir'], config['visual_query']['exp_name'])
            predictions_dir = os.path.join(exp_dir, 'predictions')
            os.makedirs(predictions_dir, exist_ok=True)
            if f'{run_id}-{f[:-4]}.json' in os.listdir(predictions_dir):
                print(f'Skipping {f[:-4]} because the predictions are already cached.')
                continue

            record_file = os.path.join(data_dir, f)
            with open(record_file, 'rb') as f:
                record = pickle.load(f)
            for annotation in record['annotations']:
                self.data.append((record_file, annotation))
        if len(self.data) == 0:
            print('All predictions are already cached.')
            exit()

        self.query_encoder = QueryEncoder(config=config)
        self.video_encoder = VideoEncoder(config=config)
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record_file, annotations = self.data[idx]
        with open(record_file, 'rb') as f:
            record = pickle.load(f)
        video_uid = record['video_uid'].decode('utf-8')
        clip_uid = record['clip_uid'].decode('utf-8')
        qset_id = annotations['query_set']

        # Add a dummy response track for test set
        if 'response_track' not in annotations:
            annotations['response_track'] = [{'frame_number': -1, 'x': -1, 'y': -1, 'width': -1, 'height': -1,
                                              'rotation': -1, 'original_width': -1, 'original_height': -1,
                                              'video_frame_number': -1}]

        # Get query info
        query_encoder_out = self.query_encoder(record_file, annotations)
        query_tokens = query_encoder_out['query_tokens']
        query_bboxes = query_encoder_out['query_bboxes']
        query_frame_numbers = query_encoder_out['query_frame_numbers']
        query_crops = query_encoder_out['query_crops']
        query_timestep = annotations['query_frame']

        # Get objects info
        video_encoder_out = self.video_encoder(record_file, annotations, query_tokens, 0.6)
        object_tokens = video_encoder_out['object_tokens']
        object_points = video_encoder_out['object_points']
        object_attn_mask = torch.full(object_tokens.size()[:1], 1, dtype=torch.float32)
        frame_ids = video_encoder_out['frame_ids']
        frames = video_encoder_out['frames']

        # Get target response track
        response_track = []
        for track in annotations['response_track']:
            frame_number = track['frame_number']
            x = int((track['x'] / track['original_width']) * frames.shape[2])
            y = int((track['y'] / track['original_height']) * frames.shape[1])
            height = int((track['height'] / track['original_height']) * frames.shape[1])
            width = int((track['width'] / track['original_width']) * frames.shape[2])
            response_track.append([frame_number, x, y, width, height])
        response_track = torch.tensor(response_track)

        item = {
            'object_tokens': object_tokens,
            'object_points': object_points,
            'object_attn_mask': object_attn_mask,
            'query_tokens': query_tokens,
            'query_bboxes': query_bboxes,
            'query_frame_numbers': query_frame_numbers,
            'query_crops': query_crops,
            'query_timestep': query_timestep,
            'frame_ids': frame_ids,
            'frames': frames,
            'response_track': response_track,
            'video_uid': video_uid,
            'clip_uid': clip_uid,
            'qset_id': qset_id,
        }
        return item