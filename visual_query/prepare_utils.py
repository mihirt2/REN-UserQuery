import os
import subprocess
import argparse
import numpy as np
import skvideo.io
import concurrent.futures
import shutil
import requests
import time


def create_base_parser(dataset_name):
    parser = argparse.ArgumentParser(description=f'Creates the records for the {dataset_name} dataset')
    parser.add_argument(
        '--fold',
        dest='fold',
        default=0,
        type=int,
        help='Integer id to select a fold. Each fold creates a separate file. Used for parallization.'
    )
    parser.add_argument(
        '--num_folds',
        dest='num_folds',
        default=1,
        type=int,
        help='Total number of folds',
    )
    parser.add_argument(
        '--seed',
        dest='seed',
        default=1337,
        type=int,
        help='Random seed used for shuffling data'
    )
    parser.add_argument(
        '--split',
        dest='split',
        default='train',
        type=str,
        help='Split of the data to use'
    )
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        type=str,
        help='Directory to save the processed data'
    )
    parser.add_argument(
        '--data_dir',
        dest='data_dir',
        type=str,
        help='Image directory.'
    )
    return parser


def get_size_for_resize(image_size, shorter_size_trg=384, longer_size_max=512):
    w, h = image_size
    size = shorter_size_trg

    if min(w, h) <= size:
        return w, h

    min_original_size = float(min((w, h)))
    max_original_size = float(max((w, h)))
    if max_original_size / min_original_size * size > longer_size_max:
        size = int(round(longer_size_max * min_original_size / max_original_size))

    if (w <= h and w == size) or (h <= w and h == size):
        return w, h
    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)
    return ow, oh


def pad(data, pad_width, pad_value):
    if pad_width == 'maxlen':
        pad_width = 0
        for item in data:
            pad_width = max(pad_width, len(item))
    
    padded_data = []
    for item in data:
        if len(item) > pad_width:
            item = item[:pad_width]
        else:
            while len(item) < pad_width:
                item.append(pad_value)
        padded_data.append(item)
    return padded_data


def download_video(url, save_dir):
    filename = url.split('/')[-1]
    video_path = os.path.join(save_dir, filename)
    if os.path.exists(video_path):
        if video_path.endswith('.mkv'):
            video_path = f'{video_path[:-4]}.mp4'
        if video_path.endswith('.webm'):
            video_path = f'{video_path[:-5]}.mp4'
        print(f'Video downloaded at: {video_path}')
        return video_path
    
    try:
        with requests.get(url, stream=True) as response:
            with open(video_path, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
        if video_path.endswith('.mkv'):
            cmd = f'ffmpeg -loglevel error -i {video_path} -codec copy {video_path[:-4]}.mp4'
            subprocess.run(cmd, shell=True, check=True)
            video_path = f'{video_path[:-4]}.mp4'
        if video_path.endswith('.webm'):
            cmd = f'ffmpeg -loglevel error -i {video_path} -codec copy {video_path[:-5]}.mp4'
            subprocess.run(cmd, shell=True, check=True)
            video_path = f'{video_path[:-5]}.mp4'
        print(f'Video downloaded to: {video_path}')
        return video_path
    
    except Exception as e:
        print(f'Error downloading video: {e}')


def detect_black_bars_from_video(frames, blackbar_threshold=16, max_perc_to_trim=.2):
    has_content = frames.max(axis=(0, -1)) >= blackbar_threshold
    h, w = has_content.shape

    y_frames = np.where(has_content.any(1))[0]
    if y_frames.size == 0:
        print("Oh no, there are no valid yframes")
        y_frames = [h // 2]

    y1 = min(y_frames[0], int(h * max_perc_to_trim))
    y2 = max(y_frames[-1] + 1, int(h * (1 - max_perc_to_trim)))

    x_frames = np.where(has_content.any(0))[0]
    if x_frames.size == 0:
        print("Oh no, there are no valid xframes")
        x_frames = [w // 2]
    x1 = min(x_frames[0], int(w * max_perc_to_trim))
    x2 = max(x_frames[-1] + 1, int(w * (1 - max_perc_to_trim)))
    return y1, y2, x1, x2


def extract_single_frame_from_video(video_file, t, verbosity=0):
    timecode = '{:.3f}'.format(t)
    reader = skvideo.io.FFmpegReader(video_file,
                                     inputdict={'-ss': timecode, '-threads': '1'},
                                     outputdict={'-r': '1', '-q:v': '2', '-pix_fmt': 'rgb24', '-frames:v': '1'},
                                     verbosity=verbosity)
    try:
        frame = next(iter(reader.nextFrame()))
    except StopIteration:
        frame = None
    return frame


def extract_frames_from_video(video_file, times, use_multithreading=False, use_rgb=True,
                              blackbar_threshold=32, max_perc_to_trim=.20, verbose=False):
    def _extract(i):
        return i, extract_single_frame_from_video(video_file, times[i], verbosity=10 if verbose else 0)

    start = time.time()

    if not use_multithreading:
        frames = [_extract(i)[1] for i in range(len(times))]
    else:
        frames = [None for t in times]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            submitted_threads = (executor.submit(_extract, i) for i in range(len(times)))
            for future in concurrent.futures.as_completed(submitted_threads):
                try:
                    i, img = future.result()
                    frames[i] = img
                except Exception as exc:
                    print("{}".format(str(exc)), flush=True)
    if verbose:
        print("Extracting frames from video, multithreading={} took {:.3f}".format(use_multithreading,
                                                                                   time.time() - start), flush=True)
    if any([x is None for x in frames]):
        print(f"Fail on {video_file}", flush=True)
        return None

    frames = np.stack(frames)
    y1, y2, x1, x2 = detect_black_bars_from_video(frames, blackbar_threshold=blackbar_threshold,
                                                  max_perc_to_trim=max_perc_to_trim)
    frames = frames[:, y1:y2, x1:x2]
    return frames