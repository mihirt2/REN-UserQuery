import os
import json
import tqdm
import argparse
from collections import defaultdict


def collate_predictions(predictions_dir, test_annotation):
    predictions = None

    # Read all the cached json files for each clip and prepare a combined prediction dict
    for video_prediction_file in os.listdir(predictions_dir):
        video_prediction_path = os.path.join(predictions_dir, video_prediction_file)
        with open(video_prediction_path, 'r') as f:
            video_predictions = json.load(f)
        
        if predictions is None:
            predictions = video_predictions
        else:
            predictions['results']['videos'].append(video_predictions['results']['videos'][0])

    # Normalize predictions
    for video in predictions['results']['videos']:
        video_uid = video['video_uid']
        frame_height, frame_width = video['frame_height'], video['frame_width']
        for clip in video['clips']:
            clip_uid = clip['clip_uid']
            for clip_predictions in clip['predictions']:
                for qset_id in clip_predictions['query_sets']:
                    
                    original_width, original_width = None, None
                    video_annotations = test_annotation['videos']
                    for vannot in video_annotations:
                        if vannot['video_uid'] == video_uid:
                            for cannot in vannot['clips']:
                                if cannot['clip_uid'] == clip_uid:
                                    for annotation in cannot['annotations']:
                                        qannot = annotation['query_sets'][qset_id]
                                        if qannot['is_valid']:
                                            original_width = qannot['visual_crop']['original_width']
                                            original_height = qannot['visual_crop']['original_height']

                    for bbox in clip_predictions['query_sets'][qset_id]['bboxes']:
                        bbox['x1'] = int((bbox['x1'] / frame_width) * original_width)
                        bbox['y1'] = int((bbox['y1'] / frame_height) * original_height)
                        bbox['x2'] = int((bbox['x2'] / frame_width) * original_width)
                        bbox['y2'] = int((bbox['y2'] / frame_height) * original_height)

    # Combine predictions with the same video_uid
    merged_video_predictions = defaultdict(lambda: {'clips': []})
    for video in predictions['results']['videos']:
        video_uid = video['video_uid']
        merged_video_predictions[video_uid]['clips'].extend(video['clips'])
    merged_video_predictions = [{'video_uid': video_uid, **data} 
                                for video_uid, data in merged_video_predictions.items()]
    predictions['results']['videos'] = merged_video_predictions

    return predictions


def order_predictions(model_predictions, test_annotations):
    video_annotations = test_annotations['videos']
    video_predictions = model_predictions['results']['videos']
    rearranged_video_predictions = []

    # Match the video_uid order in predictions with the order in annotations
    for video_annotation in video_annotations:
        video_uid = video_annotation['video_uid']
        video_prediction = None
        for vpred in video_predictions:
            if vpred['video_uid'] == video_uid:
                video_prediction = vpred
                break 
        if video_prediction is None:
            print(f'[INFO] Adding an empty prediction for video {video_uid}')
            video_prediction = {'video_uid': video_uid, 'clips': []}

        # Match the clip_uid order in video_prediction with the order in video_annotation
        rearranged_clip_predictions = []
        for clip_annotation in video_annotation['clips']:
            clip_uid = clip_annotation['clip_uid']
            clip_prediction = None
            for cpred in video_prediction['clips']:
                if cpred['clip_uid'] == clip_uid:
                    clip_prediction = cpred
                    break
            if clip_prediction is None:
                print(f'[INFO] Adding a dummy prediction for clip {clip_uid}')
                clip_prediction = {'clip_uid': clip_uid, 'predictions': []}
                for annotation in clip_annotation['annotations']:
                    dummy_prediction = {'query_sets': {}}
                    for qset_id in annotation['query_sets']:
                        dummy_bbox = {'fno': 0, 'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}
                        dummy_prediction['query_sets'][qset_id] = {'bboxes': [dummy_bbox], 'score': 0.0}
                    clip_prediction['predictions'].append(dummy_prediction)

            # Handle missing qset ids
            for clip_annot, clip_pred in zip(clip_annotation['annotations'], clip_prediction['predictions']):
                for qset_id in clip_annot['query_sets']:
                    if not clip_annot['query_sets'][qset_id]['is_valid']:
                        continue
                    if qset_id not in clip_pred['query_sets']:
                        print(f'[INFO] Adding an dummy prediction for query set {qset_id} in clip {clip_uid}')
                        dummy_bbox = {'fno': 0, 'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}
                        clip_pred['query_sets'][qset_id] = {'bboxes': [dummy_bbox], 'score': 0.0}

            rearranged_clip_predictions.append(clip_prediction)
        
        video_prediction['clips'] = rearranged_clip_predictions
        rearranged_video_predictions.append(video_prediction)
    model_predictions['results']['videos'] = rearranged_video_predictions
    return model_predictions


def validate_model_predictions(model_predictions, test_annotations):
    assert type(model_predictions) == type({})
    for key in ['version', 'challenge', 'results']:
        assert key in model_predictions.keys()
    assert model_predictions['version'] == '1.0.5'
    assert model_predictions['challenge'] == 'ego4d_vq2d_challenge'
    assert type(model_predictions['results']) == type({})
    assert 'videos' in model_predictions['results']

    video_annotations = test_annotations['videos']
    video_predictions = model_predictions['results']['videos']
    assert len(video_predictions) == len(video_annotations)

    n_samples = 0
    for v in video_annotations:
        for c in v['clips']:
            for a in c['annotations']:
                for _, q in a['query_sets'].items():
                    if q['is_valid']:
                        n_samples += 1

    pbar = tqdm.tqdm(total=n_samples, desc='Validating user predictions')
    for vannot, vpred in zip(video_annotations, video_predictions):
        assert type(vpred) == type({})
        for key in ['video_uid', 'clips']:
            assert key in vpred
        assert vannot['video_uid'] == vpred['video_uid']
        assert type(vpred['clips']) == type([])
        assert len(vannot['clips']) == len(vpred['clips'])

        for cpreds in vpred['clips']:
            assert type(cpreds) == type({})
            for key in ['clip_uid', 'predictions']:
                assert key in cpreds
        
        clips_annots = vannot['clips']
        clips_preds = vpred['clips']
        for clip_annots, clip_preds in zip(clips_annots, clips_preds):
            assert clip_annots['clip_uid'] == clip_preds['clip_uid']
            assert type(clip_preds['predictions']) == type([])
            assert len(clip_preds['predictions']) == len(clip_annots['annotations'])
            
            for clip_annot, clip_pred in zip(clip_annots['annotations'], clip_preds['predictions']):
                assert type(clip_pred) == type({})
                assert 'query_sets' in clip_pred
                valid_query_set_annots = {
                    k: v
                    for k, v in clip_annot['query_sets'].items()
                    if v['is_valid']
                }
                valid_query_set_preds = {
                    k: v
                    for k, v in clip_pred['query_sets'].items()
                    if clip_annot['query_sets'][k]['is_valid']
                }
                assert set(list(valid_query_set_preds.keys())) == set(list(valid_query_set_annots.keys()))

                for _, qset in clip_pred['query_sets'].items():
                    assert type(qset) == type({})
                    for key in ['bboxes', 'score']:
                        assert key in qset
                    pbar.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_unannotated_path', type=str, required=True,
                        help='Path to vq_test_unannotated.json.')
    parser.add_argument('--predictions_dir', type=str, required=True,
                        help='Path to the directory containing predictions.')
    args = parser.parse_args()

    test_unannotated_path = args.test_unannotated_path
    predictions_dir = args.predictions_dir
    predictions_save_path = 'predictions.json'
    with open(test_unannotated_path, 'r') as f:
        test_annotations = json.load(f)

    collated_predictions = collate_predictions(predictions_dir, test_annotations)
    model_predictions = order_predictions(collated_predictions, test_annotations)

    validate_model_predictions(model_predictions, test_annotations)

    with open(predictions_save_path, 'w') as f:
        json.dump(model_predictions, f)
    print(f'Saved predictions json at {predictions_save_path}')