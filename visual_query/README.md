# Visual Query Localization with REN

This folder contains code for evaluating REN on the **Visual Query Localization (VQL)** task using the [Ego4D VQ2D benchmark](https://ego4d-data.org/tasks/vq2d/).

Our pipeline follows a multi-stage procedure:

1. **Region Extraction**: Use REN to extract region tokens from multi-resolution crops of video frames and the visual query.
2. **Candidate Matching**: Compute cosine similarity to select candidate regions in the video that match the query.
3. **Bounding Box Estimation**: Convert the point prompt of selected candidates into bounding boxes using SAM 2, and refine the candidates by cropping around them.
4. **Initial Tracking**: Use SAM 2 to track the last selected region across frames.
5. **Query Augmentation**: Generate new visual queries from the initial track using REN and repeat the matching, refinement, and tracking steps.

This pipeline builds upon the method proposed in [RELOCATE: A Simple Training-Free Baseline for Visual Query Localization Using Region-Based Representations](https://arxiv.org/abs/2412.01826), but replaces SAM-based region token pooling with REN. This substitution results in a **60× speed-up**, allowing us to efficiently process multi-resolution crops while maintaining higher accuracy.

REN achieves **state-of-the-art performance** on the Ego4D VQ2D benchmark. For detailed results, refer to the [official leaderboard](https://eval.ai/web/challenges/challenge-page/1843/leaderboard/4326).


## Dataset Download
Follow the [Ego4D instructions](https://ego4d-data.org/docs/start-here/) to download the full scale videos and annotations.

Note: The dataset is large, with a total download size of ~9 TB.


## Data Preprocessing
We preprocess the videos into pickle files using `prepare.py`:
```
python prepare.py --split val --data_dir /path/to/vq2d/ --save_dir /path/to/save/vq2d/records/
```
The preprocessed records require 2.3 TB for the training set, 703 GB for the validation set, and 722 GB for the test set.

**Note:** Since this is a training-free approach, you only need to download and process the specific data split you want to evaluate (e.g., val or test).

Update the `config.yaml` with the path to the preprocessed records:
```
train_data_dir: '/path/to/vq2d/records/train/'
val_data_dir: '/path/to/vq2d/records/val/'
test_data_dir: '/path/to/vq2d/records/test/'
```

## Running Eval
Set `split` and `exp_name` under the `visual_query` scope in `config.yaml`, then run:
```
bash run.sh
```
This will sequentially process each video in the specified split and cache the predictions as JSON files in `logs/<exp_name>/predictions/`. If you have multiple GPUs, consider parallelizing this step—e.g., by assigning one GPU per video—to speed up processing.

If running on the train or validation set, metrics can be computed using:
```
python eval.py
```

If running on the test split, you can prepare the submission file for the evaluation server by running:
```
python submit.py --test_unannotated_path /path/to/vq_test_unannotated.json --predictions_dir logs/<exp_name>/predictions/
```