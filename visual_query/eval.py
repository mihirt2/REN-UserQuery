import os
from tqdm import tqdm
import yaml
import json
from metrics import BBox, ResponseTrack, compute_visual_query_metrics
from vq_utils import print_log


class Evaluator():
    def collate(self, predictions_dir, exp_dir):
        predictions = None
        video_prediction_files = sorted(os.listdir(predictions_dir), key=lambda x: int(x.split('-')[0]))
        for video_prediction_file in tqdm(video_prediction_files):
            video_prediction_path = os.path.join(predictions_dir, video_prediction_file)
            with open(video_prediction_path, 'r') as f:
                video_predictions = json.load(f)
            
            if predictions is None:
                predictions = video_predictions
            else:
                predictions['results']['videos'].append(video_predictions['results']['videos'][0])
            
        save_path = os.path.join(exp_dir, 'collated_prediction.json')
        with open(save_path, 'w') as f:
            json.dump(predictions, f)
        return save_path

    def run(self, predictions_path):
        with open(predictions_path, 'r') as f:
            predictions = json.load(f)

        results = predictions['results']
        predicted_response_track, ground_truth_response_track, visual_crop_boxes = [], [], []
        for video_results in results['videos']:
            for clip_results in video_results['clips']:
                for predictions in clip_results['predictions']:
                    for qset_id in predictions['query_sets']:
                        if '-gt' in qset_id:
                            continue
                        prediction = predictions['query_sets'][qset_id]
                        ground_truth = predictions['query_sets'][qset_id + '-gt']

                        # Create predicted response track
                        predicted_bboxes = [BBox(pt['fno'], pt['x1'], pt['y1'], pt['x2'], pt['y2'])
                                            for pt in prediction['bboxes']]
                        predicted_score = prediction['score']
                        predicted_response_track.append([ResponseTrack(predicted_bboxes, predicted_score)])

                        # Create ground-truth response track
                        ground_truth_bboxes = [BBox(pt['fno'], pt['x1'], pt['y1'], pt['x2'], pt['y2'])
                                               for pt in ground_truth['bboxes']]
                        ground_truth_response_track.append(ResponseTrack(ground_truth_bboxes))

                        # Add a dummy visual crop bbox for evaluation
                        visual_crop_boxes.append(BBox(-1, -1, -1, -1, -1))

        metrics = compute_visual_query_metrics(predicted_response_track, ground_truth_response_track,
                                               visual_crop_boxes)
        return metrics


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    exp_dir = os.path.join(config['visual_query']['save_dir'], config['visual_query']['exp_name'])
    predictions_dir = os.path.join(exp_dir, 'predictions')
    print(f'Evaluating cached predictions from {predictions_dir}')

    evaluator = Evaluator()
    prediction_save_path = evaluator.collate(predictions_dir, exp_dir)
    if config['visual_query']['split'] != 'test':
        metrics = evaluator.run(prediction_save_path)
        for area_range, computed_metrics in metrics.items():
            print_log(f'Area range: {area_range}', exp_dir)
            for metric_name, metric_value in computed_metrics.items():
                print_log(f'  {metric_name} \t-\t {metric_value:.4f}', exp_dir)