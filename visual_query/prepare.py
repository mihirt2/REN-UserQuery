import os
import numpy as np
import json
from PIL import Image
import pickle
from tqdm import tqdm
import prepare_utils as utils


def make_dataset(data):
    videos = data['videos']
    clips = []
    for i in range(len(videos)):
        for clip in videos[i]['clips']:
            annotation = {k: clip[k] for k in clip}
            annotation['video_uid'] = videos[i]['video_uid']
            clips.append(annotation)
    return np.array(clips)


def create_feature_dict(item, data_dir):
    video_uid = item['video_uid']
    clip_uid = item['clip_uid']
    fps = item['clip_fps']
    video_file = os.path.join(data_dir, f'v2/full_scale/{video_uid}.mp4')
    timesteps = np.arange(max(0.0, item['video_start_sec']), item['video_end_sec'], step=1.0 / fps)

    # Extract video frames
    frames = utils.extract_frames_from_video(video_file=video_file, times=np.unique(timesteps), use_multithreading=True)
    if frames is None:
        print('Could not extract frames from video annotation {}'.format(video_uid), flush=True)
        return None
    target_size = utils.get_size_for_resize((frames.shape[2], frames.shape[1]), shorter_size_trg=288, longer_size_max=512)
    resized_frames = []
    for frame in frames:
        image = Image.fromarray(frame, mode='RGB')
        if target_size != image.size:
            image = image.resize(target_size, resample=Image.BICUBIC)
        resized_frames.append(np.array(image))
    frames = np.array(resized_frames)

    # Extract annotations
    annotations = []
    for annotation in item['annotations']:
        for qset_id, qset in annotation['query_sets'].items():
            if not qset['is_valid']:
                continue

            visual_crop = qset['visual_crop']
            query_frame = frames[visual_crop['frame_number']]
            x = int((visual_crop['x'] / visual_crop['original_width']) * frames.shape[2])
            y = int((visual_crop['y'] / visual_crop['original_height']) * frames.shape[1])
            height = int((visual_crop['height'] / visual_crop['original_height']) * frames.shape[1])
            width = int((visual_crop['width'] / visual_crop['original_width']) * frames.shape[2])
            query_object = query_frame[y:(y + height), x:(x + width)]

            curr_annotation = {
                'metadata': {
                    'video_uid': video_uid,
                    'video_start_sec': item['video_start_sec'],
                    'video_end_sec': item['video_end_sec'],
                    'clip_fps': fps,
                },
                'clip_uid': clip_uid,
                'query_set': qset_id,
                'query_object': query_object,
                'query_frame': qset['query_frame'],
                'response_track': qset['response_track'],
                'visual_crop': qset['visual_crop'],
                'object_title': qset['object_title'],
            }
            annotations.append(curr_annotation)

    # Create feature_dict
    feature_dict = {}
    feature_dict['clip_uid'] = clip_uid.encode()
    feature_dict['video_uid'] = video_uid.encode()
    feature_dict['frames'] = frames
    feature_dict['num_frames'] = frames.shape[0]
    feature_dict['frame_height'] = frames.shape[1]
    feature_dict['frame_width'] = frames.shape[2]
    feature_dict['frame_channel'] = frames.shape[3]
    feature_dict['annotations'] = np.array(annotations)
    return feature_dict


def main():
    parser = utils.create_base_parser('vq2d')
    args = parser.parse_args()

    # Setup the info for this split
    out_dir = os.path.join(args.save_dir, f'{args.split}')
    os.makedirs(out_dir, exist_ok=True)
    if args.split == 'train':
        annotations_path = os.path.join(args.data_dir, 'v2/annotations/vq_train.json')
    elif args.split == 'val':
        annotations_path = os.path.join(args.data_dir, 'v2/annotations/vq_val.json')
    elif args.split == 'test':
        annotations_path = os.path.join(args.data_dir, 'v2/annotations/vq_test_unannotated.json')
    else:
        raise ValueError('Incorrect split mentioned.')

    # Preprocess the data
    with open(annotations_path, 'r+') as f:
        annotations = json.load(f)
    print(f'Loaded the annotations.')

    # Process and write the examples
    data = make_dataset(annotations)
    num_examples = len(data)
    dropped_examples = []
    for item in tqdm(data, desc=f'Processing {num_examples} clips', total=num_examples):
        clip_uid = item['clip_uid']
        if f'{clip_uid}.pkl' in os.listdir(out_dir):
            print(f'{clip_uid} already generated.')
            dropped_examples.append(item)
            continue
        
        feature_dict = create_feature_dict(item, args.data_dir)
        if feature_dict is None:
            dropped_examples.append(item)
            continue
    
        clip_uid = feature_dict['clip_uid'].decode('utf-8')
        with open(os.path.join(out_dir, f'{clip_uid}.pkl'), 'wb') as f:
            pickle.dump(feature_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved the processed data in {out_dir} (dropped {len(dropped_examples)} examples).')


if __name__ == '__main__':
    main()