import numpy as np
import pandas as pd
from collections import OrderedDict
from typing import Dict, List, Sequence, Tuple


METRIC_FNS = [
    lambda gt, pred: SpatioTemporalDetection(gt, pred).get_metrics(),
    lambda gt, pred: TemporalDetection(gt, pred).get_metrics(),
    lambda gt, pred: TrackingMetrics(gt, pred, ignore_iou_averaging=True).get_metrics(),
    lambda gt, pred: SuccessMetrics(gt, pred, ignore_iou_averaging=True).get_metrics(),
]
PRINT_FORMAT = '{:<30s} {:<15s}'


class BBox:
    def __init__(self, fno, x1, y1, x2, y2):
        self.fno = fno
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def area(self):
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def __repr__(self):
        return f'BBox[fno = {self.fno}, x1 = {self.x1}, y1 = {self.y1}, x2 = {self.x2}, y2 = {self.y2}]'

    def to_json(self):
        return {'fno': int(self.fno), 'x1': int(self.x1), 'x2': int(self.x2), 'y1': int(self.y1), 'y2': int(self.y2)}

    @staticmethod
    def from_json(data):
        return BBox(data['fno'], data['x1'], data['y1'], data['x2'], data['y2'])


class ResponseTrack:
    def __init__(self, bboxes: List[BBox], score: float = None):
        self._bboxes = sorted(bboxes, key=lambda x: x.fno)
        if self._check_empty(self._bboxes):
            self._empty_init()
        else:
            self._non_empty_init(self._bboxes)
        self._score = score
        self._check_contiguous()

    @property
    def temporal_extent(self):
        return (self._t_start, self._t_end)

    @property
    def bboxes(self):
        return self._bboxes

    @property
    def length(self):
        return self._length

    @property
    def score(self):
        return self._score

    def has_score(self):
        return self._score is not None

    def _check_empty(self, bboxes):
        return len(bboxes) == 0

    def _empty_init(self):
        self._t_start = 0
        self._t_end = -1
        self._length = 0
        print('Encountered empty track')

    def _non_empty_init(self, bboxes):
        self._t_start = bboxes[0].fno
        self._t_end = bboxes[-1].fno
        self._length = len(bboxes)

    def _check_contiguous(self):
        if self._length != (self._t_end - self._t_start + 1):
            raise ValueError(f'Bounding boxes that make up the response track are not contiguous')

    def __repr__(self):
        return ('ResponseTrack {\n\t' + '\n\t'.join([bbox.__repr__() for bbox in self._bboxes]) + '}')

    def volume(self):
        v = 0.0
        for bbox in self._bboxes:
            v += bbox.area()
        return v

    def to_json(self):
        score = self._score
        if score is not None:
            score = float(score)
        return {'bboxes': [bbox.to_json() for bbox in self._bboxes], 'score': score}

    @staticmethod
    def from_json(data):
        return ResponseTrack([BBox.from_json(bbox) for bbox in data['bboxes']], data['score'])


class SpatioTemporalDetection(object):
    """
    Modified from ActivityNet evaluation toolkit: https://github.com/activitynet/ActivityNet.
    """
    def __init__(self, ground_truth: List[ResponseTrack], prediction: List[List[ResponseTrack]],
                 ignore_iou_averaging: bool = False):
        self.metric_uuid = 'Spatio-temporal AP'
        self.iou_thresholds = np.array([0.25, 0.5, 0.75, 0.95])
        self.ious_to_report = [0.25]
        self.ap = None
        self.ground_truth = self._import_ground_truth(ground_truth)
        self.prediction = self._import_prediction(prediction)
        self.ignore_iou_averaging = ignore_iou_averaging

    def _import_ground_truth(self, ground_truth: List[ResponseTrack]):
        video_ids, response_tracks = [], []
        for i, gt in enumerate(ground_truth):
            video_ids.append(i)
            response_tracks.append(gt)
        ground_truth = pd.DataFrame({
            'video_id': video_ids, 
            'response_track': response_tracks
        })
        return ground_truth

    def _import_prediction(self, prediction: List[List[ResponseTrack]]):
        video_ids, response_tracks, scores = [], [], []
        for i, preds in enumerate(prediction):
            for pred in preds:
                score = pred.score
                video_ids.append(i)
                response_tracks.append(pred)
                scores.append(score)
        prediction = pd.DataFrame({
            'video_id': video_ids,
            'response_track': response_tracks,
            'score': scores
        })
        return prediction

    def evaluate(self):
        self.ap = compute_stap_detection(self.ground_truth, self.prediction, self.iou_thresholds)
        self.average_ap = self.ap.mean().item()

    def get_metrics(self):
        self.evaluate()
        metrics = OrderedDict()
        avg_suffix = f'@ IoU={self.iou_thresholds[0]:.2f}:{self.iou_thresholds[-1]:.2f}'

        if not self.ignore_iou_averaging:
            metrics[PRINT_FORMAT.format(self.metric_uuid, avg_suffix)] = self.average_ap

        for tiou_idx, tiou_thr in enumerate(self.iou_thresholds):
            if tiou_thr not in self.ious_to_report:
                continue
            metrics[PRINT_FORMAT.format(self.metric_uuid, f'@ IoU={tiou_thr:.2f}')] = self.ap[tiou_idx].item()
        return metrics


class TemporalDetection(object):
    def __init__(self, ground_truth: List[ResponseTrack], prediction: List[List[ResponseTrack]],
                 ignore_iou_averaging: bool = False):
        self.metric_uuid = 'Temporal AP'
        self.tiou_thresholds = np.array([0.25, 0.5, 0.75, 0.95])
        self.tious_to_report = [0.25]
        self.ap = None
        self.ground_truth = self._import_ground_truth(ground_truth)
        self.prediction = self._import_prediction(prediction)
        self.ignore_iou_averaging = ignore_iou_averaging

    def _import_ground_truth(self, ground_truth: List[ResponseTrack]):
        video_ids, t_start, t_end = [], [], []
        for i, gt in enumerate(ground_truth):
            video_ids.append(i)
            t_start.append(gt.temporal_extent[0])
            t_end.append(gt.temporal_extent[1])

        ground_truth = pd.DataFrame({
            'video_id': video_ids,
            't_start': t_start,
            't_end': t_end,
        })
        return ground_truth

    def _import_prediction(self, prediction: List[List[ResponseTrack]]):
        video_ids, t_start, t_end, scores = [], [], [], []
        for i, preds in enumerate(prediction):
            for pred in preds:
                assert pred.has_score()
                video_ids.append(i)
                t_start.append(pred.temporal_extent[0])
                t_end.append(pred.temporal_extent[1])
                scores.append(pred.score)
        prediction = pd.DataFrame({
            'video_id': video_ids,
            't_start': t_start,
            't_end': t_end,
            'score': scores,
        })
        return prediction

    def evaluate(self):
        self.ap = compute_tap_detection(self.ground_truth, self.prediction, self.tiou_thresholds)
        self.average_ap = self.ap.mean().item()

    def get_metrics(self):
        self.evaluate()
        metrics = OrderedDict()
        avg_suffix = (f'@ IoU={self.tiou_thresholds[0]:.2f}:{self.tiou_thresholds[-1]:.2f}')

        if not self.ignore_iou_averaging:
            metrics[PRINT_FORMAT.format(self.metric_uuid, avg_suffix)] = self.average_ap

        for tiou_idx, tiou_thr in enumerate(self.tiou_thresholds):
            if tiou_thr not in self.tious_to_report:
                continue
            metrics[PRINT_FORMAT.format(self.metric_uuid, f'@ IoU={tiou_thr:.2f}')] = self.ap[tiou_idx].item()
        return metrics


class TrackingMetrics(object):
    def __init__(self, ground_truth: List[ResponseTrack], prediction: List[List[ResponseTrack]],
                 mode: str = 'take_max_score', ignore_iou_averaging: bool = False):
        self.metric_uuid: str = 'Tracking'
        self.iou_thresholds: np.ndarray = np.array([0.5, 0.75, 0.95])
        self.ious_to_report: List[float] = [0.5]
        self.ap = None
        self.ground_truth = self._import_ground_truth(ground_truth)
        self.prediction = self._import_prediction(prediction)

        assert mode in ['take_max_stiou', 'take_max_score']
        self.mode = mode
        if mode == 'take_max_stiou':
            self.suffix = '(max iou)'
        elif mode == 'take_max_score':
            self.suffix = '(max scr)'
        self.ignore_iou_averaging = ignore_iou_averaging

    def _import_ground_truth(self, ground_truth: List[ResponseTrack]):
        video_ids, response_tracks = [], []
        for i, gt in enumerate(ground_truth):
            video_ids.append(i)
            response_tracks.append(gt)

        ground_truth = pd.DataFrame({
            'video_id': video_ids,
            'response_track': response_tracks,
        })
        return ground_truth

    def _import_prediction(self, prediction: List[List[ResponseTrack]]):
        video_ids, response_tracks, scores = [], [], []
        for i, preds in enumerate(prediction):
            for pred in preds:
                score = pred.score
                video_ids.append(i)
                response_tracks.append(pred)
                scores.append(score)
        prediction = pd.DataFrame({
            'video_id': video_ids,
            'response_track': response_tracks,
            'score': scores,
        })
        return prediction

    def evaluate(self):
        self.tracking_metrics = compute_tracking_metrics(self.ground_truth, self.prediction, self.iou_thresholds,
                                                         mode=self.mode)
        self.average_tracking_metrics = {k: v.mean().item() for k, v in self.tracking_metrics.items()}

    def get_metrics(self) -> Dict[str, float]:
        self.evaluate()
        metrics = OrderedDict()
        avg_suffix = f'@ IoU={self.iou_thresholds[0]:.2f}:{self.iou_thresholds[-1]:.2f}'

        for k, v in self.average_tracking_metrics.items():
            metric_name = '{} {} {}'.format(self.metric_uuid, k, self.suffix)

            if not self.ignore_iou_averaging:
                metrics[PRINT_FORMAT.format(metric_name, avg_suffix)] = v

            v = self.tracking_metrics[k]
            for tidx, iou_thr in enumerate(self.iou_thresholds):
                if iou_thr not in self.ious_to_report:
                    continue
                metrics[PRINT_FORMAT.format(metric_name, f'@ IoU={iou_thr:.2f}')] = v[tidx].item()
        return metrics


class SuccessMetrics(object):
    def __init__(self, ground_truth: List[ResponseTrack], prediction: List[List[ResponseTrack]],
                 mode: str = 'take_max_score', ignore_iou_averaging: bool = False):
        self.metric_uuid = 'Success'
        self.iou_thresholds = np.array([0.05, 0.1, 0.2])
        self.ious_to_report = [0.05]
        self.ap = None
        self.ground_truth = self._import_ground_truth(ground_truth)
        self.prediction = self._import_prediction(prediction)

        assert mode in ['take_max_stiou', 'take_max_score']
        self.mode = mode
        if mode == 'take_max_stiou':
            self.suffix = '(max iou)'
        elif mode == 'take_max_score':
            self.suffix = '(max scr)'
        self.ignore_iou_averaging = ignore_iou_averaging

    def _import_ground_truth(self, ground_truth: List[ResponseTrack]):
        video_ids, response_tracks = [], []
        for i, gt in enumerate(ground_truth):
            video_ids.append(i)
            response_tracks.append(gt)

        ground_truth = pd.DataFrame({
            'video_id': video_ids,
            'response_track': response_tracks,
        })
        return ground_truth

    def _import_prediction(self, prediction: List[List[ResponseTrack]]):
        video_ids, response_tracks, scores = [], [], []
        for i, preds in enumerate(prediction):
            for pred in preds:
                score = pred.score
                video_ids.append(i)
                response_tracks.append(pred)
                scores.append(score)
        prediction = pd.DataFrame({
            'video_id': video_ids,
            'response_track': response_tracks,
            'score': scores,
        })
        return prediction

    def evaluate(self):
        self.success = compute_success(self.ground_truth, self.prediction, self.iou_thresholds, mode=self.mode)
        self.average_success = self.success.mean().item()

    def get_metrics(self):
        self.evaluate()
        metrics = OrderedDict()
        avg_suffix = f'@ IoU={self.iou_thresholds[0]:.2f}:{self.iou_thresholds[-1]:.2f}'
        metric_name = '{} {}'.format(self.metric_uuid, self.suffix)

        if not self.ignore_iou_averaging:
            metrics[PRINT_FORMAT.format(metric_name, avg_suffix)] = self.average_success

        for tidx, iou_thr in enumerate(self.iou_thresholds):
            if iou_thr not in self.ious_to_report:
                continue
            metrics[PRINT_FORMAT.format(metric_name, f'@ IoU={iou_thr:.2f}')] = self.success[tidx].item()
        return metrics


def spatial_intersection(box1: BBox, box2: BBox):
    """
    Computes intersection between two bounding boxes
    """
    xi_s = max(box1.x1, box2.x1)
    xi_e = min(box1.x2, box2.x2)
    yi_s = max(box1.y1, box2.y1)
    yi_e = min(box1.y2, box2.y2)
    inter = (np.clip(xi_e - xi_s, 0, np.inf) * np.clip(yi_e - yi_s, 0, np.inf)).item()
    return inter


def spatio_temporal_iou_response_track(rt1: ResponseTrack, rt2: ResponseTrack):
    """
    Computes tube-iou between two response track windows.
    Reference: https://github.com/rafaelpadilla/review_object_detection_metrics
    """
    # Map frame numbers to boxes
    boxes1_dict = {box.fno: box for box in rt1.bboxes}
    intersection = 0.0

    # Find matching frame boxes and estimate iou
    for box2 in rt2.bboxes:
        box1 = boxes1_dict.get(box2.fno, None)
        if box1 is None:
            continue
        intersection += spatial_intersection(box1, box2)

    # Find overall volume of the two respose tracks
    volume1 = rt1.volume()
    volume2 = rt2.volume()
    iou = intersection / (volume1 + volume2 - intersection)
    return iou


def spatio_temporal_iou(target_rt: ResponseTrack, candidate_rts: List[ResponseTrack]):
    """
    Computes spatio-temporal IoU between a target response track and candidate response tracks.
    """
    ious = []
    for candidate_rt in candidate_rts:
        ious.append(spatio_temporal_iou_response_track(target_rt, candidate_rt))
    return np.array(ious)


def segment_iou(target_segment: np.ndarray, candidate_segments: np.ndarray):
    """
    Compute the temporal IoU between a target segment and all the test segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])

    # Compute intersection including non-negative overlap score
    segments_intersection = (tt2 - tt1 + 1).clip(0)

    # Compute segment union
    segments_union = ((candidate_segments[:, 1] - candidate_segments[:, 0] + 1) + \
                      (target_segment[1] - target_segment[0] + 1) - segments_intersection)

    # Compute overlap as the ratio of the intersection over union of two segments
    iou = segments_intersection.astype(float) / segments_union
    return iou


def spatial_iou(box1: BBox, box2: BBox):
    """
    Computes spatial IoU, i.e., IoU between two bounding boxes.
    """
    xi_s = max(box1.x1, box2.x1)
    xi_e = min(box1.x2, box2.x2)
    yi_s = max(box1.y1, box2.y1)
    yi_e = min(box1.y2, box2.y2)
    inter = (np.clip(xi_e - xi_s, 0, np.inf) * np.clip(yi_e - yi_s, 0, np.inf)).item()

    area1 = box1.area()
    area2 = box2.area()
    iou = inter / (area1 + area2 - inter)
    return iou


def interpolated_prec_rec(prec: np.ndarray, rec: np.ndarray):
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap


def compute_stap_detection(ground_truth: pd.DataFrame, prediction: pd.DataFrame,
                           iou_thresholds: Sequence[float] = np.linspace(0.5, 0.95, 10)):
    """
    Compute spatio-temporal average precision between ground truth and predictions data frames. If 
    multiple predictions occurs for the same predicted segment, only the one with highest score is matches 
    as true positive. This code is greatly inspired by Pascal VOC devkit.
    Args:
        ground_truth: Dataframe containing the ground truth instances
                      Required fields: ['video_id', 'response_track']
        prediction: Data frame containing the prediction instances.
                    Required fields: ['video_id, 'response_track', 'score']
        iou_thresholds: Spatio-temporal intersection over union threshold.
    Returns:
        ap: Average precision score.
    """
    ap = np.zeros(len(iou_thresholds))
    if prediction.empty:
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(iou_thresholds), len(ground_truth))) * -1

    # Sort predictions by decreasing score order
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors
    tp = np.zeros((len(iou_thresholds), len(prediction)))
    fp = np.zeros((len(iou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video_id')

    # Assign true positive to truly ground truth instances
    for idx, this_pred in prediction.iterrows():
        try:
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video_id'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = spatio_temporal_iou(this_pred['response_track'], this_gt['response_track'].values)

        # Retrieve the predictions with the highest tiou score
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(iou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue

                # Assign as true positive after the filters above
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float64)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float64)
    recall_cumsum = tp_cumsum / npos
    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(iou_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx, :], recall_cumsum[tidx, :])
    return ap


def compute_tap_detection(ground_truth: pd.DataFrame, prediction: pd.DataFrame,
                          tiou_thresholds: Sequence[float] = np.linspace(0.5, 0.95, 10)):
    """
    Compute temporatl average precision between ground truth and predictions data frames. If multiple 
    predictions occurs for the same predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.
    Args:
        ground_truth: Data frame containing the ground truth instances
                      Required fields: ['video_id', 't_start', 't_end']
        prediction: Data frame containing the prediction instances
                    Required fields: ['video_id, 't_start', 't_end', 'score']
        tiou_thresholds: Temporal intersection over union threshold
    Returns:
        ap: Average precision score
    """
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds), len(ground_truth))) * -1

    # Sort predictions by decreasing score order
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video_id')

    # Assign true positive to truly grount truth instances
    for idx, this_pred in prediction.iterrows():
        try:
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video_id'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['t_start', 't_end']].values, this_gt[['t_start', 't_end']].values)

        # Retrieve the predictions with highest tiou score
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                
                # Assign as true positive after the filters above
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float64)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float64)
    recall_cumsum = tp_cumsum / npos
    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx, :], recall_cumsum[tidx, :])
    return ap


def compute_tracking_metrics(ground_truth: pd.DataFrame, prediction: pd.DataFrame,
                             iou_thresholds: Sequence[float] = np.linspace(0.5, 0.95, 10),
                             mode: str = 'take_max_stiou'):
    """
    Compute % frames from the ground truth that were accurately recovered by the predictions.
    Args:
        ground_truth: Data frame containing the ground truth instances
                      Required fields: ['video_id', 'response_track']
        prediction: Data frame containing the prediction instances
                    Required fields: ['video_id, 'response_track', 'score']
        iou_thresholds: Spatio-temporal intersection over union threshold
    Returns:
        recovery: recovery %
    """
    assert mode in ['take_max_stiou', 'take_max_score']
    robustness = np.zeros(len(iou_thresholds))
    if prediction.empty:
        return {'% recovery': robustness}

    # Sort predictions by decreasing score order
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize the following variables
    # st_iou: Spatio-temporal IoU between the predicted and ground-truth RTs
    # iou_sum: Sum of IoUs over the accurately tracked bboxes for an RT
    # track_acc_count: Number of accurately tracked bboxes for an RT
    # track_total_count: Number of total bboxes in the ground-truth RT
    st_iou = np.zeros((len(iou_thresholds), len(prediction)))
    iou_sum = np.zeros((len(iou_thresholds), len(prediction)))
    track_acc_count = np.zeros((len(iou_thresholds), len(prediction)))
    track_total_count = np.zeros((len(iou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video_id')
    for idx, this_pred in prediction.iterrows():
        try:
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video_id'])
        except Exception as e:
            continue

        # Assuming that there is only 1 ground-truth per video
        this_gt = ground_truth_videoid.reset_index()
        stiou_arr = spatio_temporal_iou(this_pred['response_track'], this_gt['response_track'].values)[0]
        stiou_matches = spatio_temporal_iou_matches(this_pred['response_track'], this_gt['response_track'].values)[0]

        # Get accuracy and count over the accurately tracked frames
        for tidx, iou_thr in enumerate(iou_thresholds):
            stiou_values = np.array(list(stiou_matches.values()))
            mask = stiou_values >= iou_thr
            t_sum = np.count_nonzero(mask)
            i_sum = stiou_values[mask].sum().item()
            st_iou[tidx, idx] = stiou_arr
            iou_sum[tidx, idx] = i_sum
            track_acc_count[tidx, idx] = t_sum
            track_total_count[tidx, idx] = this_gt['response_track'].values[0].length

    # For each ground-truth, pick the prediction with highest st-IoU or score
    final_iou_sum = [[] for _ in iou_thresholds]
    final_track_acc_count = [[] for _ in iou_thresholds]
    final_track_total_count = [[] for _ in iou_thresholds]
    prediction_gbvn = prediction.groupby('video_id')

    for idx, this_gt in ground_truth.iterrows():
        prediction_videoid = prediction_gbvn.get_group(this_gt['video_id'])
        pred_idxs = prediction_videoid.index.tolist()

        for tidx, iou_thr in enumerate(iou_thresholds):
            if mode == 'take_max_stiou':
                max_idx = st_iou[tidx, pred_idxs].argmax().item()
            elif mode == 'take_max_score':
                max_idx = prediction_videoid['score'].values.argmax().item()

            max_idx = pred_idxs[max_idx]
            final_iou_sum[tidx].append(iou_sum[tidx, max_idx])
            final_track_acc_count[tidx].append(track_acc_count[tidx, max_idx])
            final_track_total_count[tidx].append(track_total_count[tidx, max_idx])

    for tidx, iou_thr in enumerate(iou_thresholds):
        iou_sum = np.array(final_iou_sum[tidx])
        track_acc_count = np.array(final_track_acc_count[tidx])
        track_total_count = np.array(final_track_total_count[tidx])
        if track_total_count.sum() > 0:
            robustness[tidx] = 100.0 * track_acc_count.sum() / (track_total_count.sum())
    return {'% recovery': robustness}


def spatial_matches_response_track(pred: ResponseTrack, gt: ResponseTrack):
    """
    For each bbox in gt, find a match in pred and measure the per-frame IoU.  Set IoU to zero if no 
    match is found. This function assumes that each bbox in the list corresponds to a different frame.
    """
    # Map frame numbers to boxes
    gt_dict = {box.fno: box for box in gt.bboxes}
    ious = {box.fno: 0.0 for box in gt.bboxes}

    # Find matching frame boxes and estimate iou
    for pred_box in pred.bboxes:
        gt_box = gt_dict.get(pred_box.fno, None)
        if gt_box is not None:
            ious[gt_box.fno] = spatial_iou(gt_box, pred_box)
    return ious


def spatio_temporal_iou_matches(target_rt: ResponseTrack, candidate_rts: List[ResponseTrack]):
    """
    For each bbox in each candidate response track, find the IoU between itself and a bbox from the target
    response track. In case no match is found for a particular bbox in the candidate, then the IoU is set to zero.
    """
    ious = []
    for candidate_rt in candidate_rts:
        ious.append(spatial_matches_response_track(target_rt, candidate_rt))
    return ious


def compute_success(ground_truth: pd.DataFrame, prediction: pd.DataFrame,
                    iou_thresholds: Sequence[float] = np.linspace(0.5, 0.95, 10), mode: str = 'take_max_stiou'):
    """
    Compute success %, i.e., the % of cases where there is sufficient overlap between ground truth and 
    predictions data frames.
    Args:
        ground_truth: Data frame containing the ground truth instances
                      Required fields: ['video_id', 'response_track']
        prediction: Data frame containing the prediction instances.
                    Required fields: ['video_id, 'response_track', 'score']
        iou_thresholds: Spatio-temporal intersection over union threshold
    Returns:
        success: Success %
    """
    assert mode in ['take_max_stiou', 'take_max_score']
    success = np.zeros(len(iou_thresholds))
    if prediction.empty:
        return {'Success': success}

    # Sort predictions by decreasing score order
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize the st_iou and track_success
    st_iou = np.zeros((len(iou_thresholds), len(prediction)))
    track_success = np.zeros((len(iou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video_id')

    for idx, this_pred in prediction.iterrows():
        try:
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video_id'])
        except Exception as e:
            continue
        
        # Assuming that there is only 1 ground-truth per video
        this_gt = ground_truth_videoid.reset_index()
        stiou_arr = spatio_temporal_iou(this_pred['response_track'], this_gt['response_track'].values)[0]

        # Get accuracy and count over the accurately tracked frames
        for tidx, iou_thr in enumerate(iou_thresholds):
            if stiou_arr >= iou_thr:
                track_success[tidx, idx] = 1
            st_iou[tidx, idx] = stiou_arr

    # For each ground-truth, pick the prediction with highest st-IoU or score
    final_track_success = [[] for _ in iou_thresholds]
    prediction_gbvn = prediction.groupby('video_id')
    for idx, this_gt in ground_truth.iterrows():
        prediction_videoid = prediction_gbvn.get_group(this_gt['video_id'])
        pred_idxs = prediction_videoid.index.tolist()

        for tidx, iou_thr in enumerate(iou_thresholds):
            if mode == 'take_max_stiou':
                max_idx = st_iou[tidx, pred_idxs].argmax().item()
            elif mode == 'take_max_score':
                max_idx = prediction_videoid['score'].values.argmax().item()
            
            max_idx = pred_idxs[max_idx]
            final_track_success[tidx].append(track_success[tidx, max_idx])

    for tidx, iou_thr in enumerate(iou_thresholds):
        track_success = np.array(final_track_success[tidx])
        success[tidx] = np.mean(track_success) * 100.0
    return success


def compute_visual_query_metrics(predicted_response_track: List[List[ResponseTrack]],
                                 ground_truth_response_track: List[ResponseTrack], visual_crop_boxes: List[BBox],
                                 accessed_frames_in_clip: List[int] = None, total_frames_in_clip: List[int] = None,
                                 vc_rt_pairings: Dict[str, Tuple[str, str]] = {'all': ('all', 'all')},
                                 area_ranges: Dict[str, List[int]] = {'all': [0 ** 2, 1e5 ** 2], 
                                                                      'small': [0 ** 2, 64 ** 2],
                                                                      'medium': [64 ** 2, 192 ** 2],
                                                                      'large': [192 ** 2, 1e5 ** 2]}):
    """
    Compute model performance on the visual query task. Includes the following metrics:
    Spatio-temporal AP, temporal AP, success, recovery, search efficiency
    """

    # Calculate area of the visual crops
    vc_areas = np.array([abs(vc_bbox.x2 - vc_bbox.x1) * abs(vc_bbox.y2 - vc_bbox.y1)
                         for vc_bbox in visual_crop_boxes])
    
    # Calculate max areas in the response tracks
    rt_areas = []
    for rt in ground_truth_response_track:
        area = (np.array([abs(rt_bbox.x2 - rt_bbox.x1) * abs(rt_bbox.y2 - rt_bbox.y1)
                          for rt_bbox in rt.bboxes]).max().item())
        rt_areas.append(area)
    rt_areas = np.array(rt_areas)

    # Calculate metrics for each vc_rt_pairing
    num_valid = 0
    pair_metrics = OrderedDict()
    for pair_name, (vc_cat, rt_cat) in vc_rt_pairings.items():
        vc_range = area_ranges[vc_cat]
        rt_range = area_ranges[rt_cat]

        # Get data points satifying the pairing criterion
        mask = ((vc_areas >= vc_range[0]) & (vc_areas < vc_range[1]) 
                & (rt_areas >= rt_range[0]) & (rt_areas < rt_range[1]))
        num_valid += mask.sum()

        # Ignore pairing if there are not valid data points
        if np.count_nonzero(mask) == 0:
            continue

        # Calculate metrics
        pred_rt = [predicted_response_track[i] for i, cond in enumerate(mask) if cond]
        gt_rt = [ground_truth_response_track[i] for i, cond in enumerate(mask) if cond]

        if accessed_frames_in_clip is not None:
            acc_frames = [accessed_frames_in_clip[i] for i, cond in enumerate(mask) if cond]
            tot_frames = [total_frames_in_clip[i] for i, cond in enumerate(mask) if cond]

        metrics = OrderedDict()
        for metric_fn in METRIC_FNS:
            metrics.update(metric_fn(gt_rt, pred_rt))

        if accessed_frames_in_clip is not None and len(acc_frames) > 0:
            metrics['Search efficiency (%)'] = (1 - np.array(acc_frames).astype(np.float32) \
                                                / np.array(tot_frames)).mean() * 100.0
        pair_metrics[pair_name] = metrics
    return pair_metrics