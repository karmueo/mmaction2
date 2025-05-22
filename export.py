import argparse
from mmengine import Config, DictAction
from mmaction.apis import init_recognizer
import torch
import torch.nn as nn
import onnx
import onnxruntime
from mmengine.dataset import Compose, pseudo_collate
from typing import List, Optional, Tuple, Union
from mmaction.structures import ActionDataSample
from mmengine.registry import init_default_scope
import os.path as osp
import numpy as np
import cv2


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def preprocess_frames_for_onnx(folder_path, num_frames=16, target_size=(224, 224)):
    """
    Reads image frames from a specified folder, normalizes them,
    and prepares them as a NumPy array for ONNX model input.

    Args:
        folder_path (str): The path to the folder containing the image frames.
                           Images should be named 0.jpg, 1.jpg, ..., i.jpg.
        num_frames (int): The number of frames to process (N in NTCHW).
        target_size (tuple): The target height and width for resizing (H, W).

    Returns:
        numpy.ndarray: The processed frames in NTCHW format (1, N, C, H, W),
                       or None if an error occurs.
    """
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    processed_frames = []

    for i in range(num_frames):
        img_path = osp.join(folder_path, f"{i}.jpg")

        if not osp.exists(img_path):
            print(f"Warning: Image {img_path} not found. Stopping.")
            # Depending on requirements, you might want to pad or raise an error.
            # For this example, we'll use the frames found so far if any,
            # or return None if no frames were processed to meet the num_frames requirement.
            if not processed_frames:  # if no frames were processed yet
                print(
                    f"Error: Could not read enough frames. Expected {num_frames}, found {i}.")
                return None
            break  # Stop if a frame is missing

        # Read the image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            continue

        # Resize the image
        # (H, W, C) - OpenCV default is BGR
        if img.shape[0] != target_size[0] or img.shape[1] != target_size[1]:
            # Resize to target size
            img_resized = cv2.resize(img, target_size)
        else:
            img_resized = img

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # Convert to float32
        img_float = img_rgb.astype(np.float32)

        # Normalize
        normalized_img = (img_float - mean) / std

        # Transpose from HWC to CHW (Channels, Height, Width)
        img_chw = normalized_img.transpose((2, 0, 1))  # C, H, W

        processed_frames.append(img_chw)

    if not processed_frames:
        print("Error: No frames were processed.")
        return None

    # Stack frames to form a sequence (N, C, H, W)
    # If fewer than num_frames were loaded due to missing files,
    # this will create a stack with the available frames.
    # You might need to add padding here if your model strictly expects num_frames.
    if len(processed_frames) < num_frames:
        print(
            f"Warning: Loaded {len(processed_frames)} frames, but expected {num_frames}.")
        # Example: Pad with zeros if necessary (this is a simple padding strategy)
        # while len(processed_frames) < num_frames:
        #     pad_frame = np.zeros((3, target_size[0], target_size[1]), dtype=np.float32)
        #     processed_frames.append(pad_frame)
        # For this example, we will proceed with the frames we have,
        # but the user should be aware if the count is less than num_frames.

    video_data = np.stack(processed_frames, axis=0)  # N, C, H, W

    # Add batch dimension (1, N, C, H, W)
    video_data_batch = np.expand_dims(video_data, axis=0)  # 1, N, C, H, W

    # Ensure the N dimension matches num_frames if strict matching is needed.
    # If padding was done, this check might be redundant.
    # If not, and fewer frames were loaded, this will reflect the actual number of loaded frames.
    current_n = video_data_batch.shape[1]
    if current_n != num_frames:
        print(
            f"Info: The final array has {current_n} frames instead of the initially requested {num_frames} due to missing files or processing issues.")
        # If you need to force the shape to (1, num_frames, C, H, W) and handle mismatches:
        # Option 1: If current_n < num_frames, you might pad here.
        # Option 2: If current_n > num_frames (e.g., if loop logic was different), you might truncate.
        # For this implementation, we are just informing the user.

    return video_data_batch.astype(np.float32)  # Ensure float32 for ONNX


def inference_recognizer(model: nn.Module,
                         video: Union[str, dict],
                         test_pipeline: Optional[Compose] = None
                         ):
    """Inference a video with the recognizer.

    Args:
        model (nn.Module): The loaded recognizer.
        video (Union[str, dict]): The video file path or the results
            dictionary (the input of pipeline).
        test_pipeline (:obj:`Compose`, optional): The test pipeline.
            If not specified, the test pipeline in the config will be
            used. Defaults to None.

    Returns:
        :obj:`ActionDataSample`: The inference results. Specifically, the
        predicted scores are saved at ``result.pred_score``.
    """

    if test_pipeline is None:
        cfg = model.cfg
        init_default_scope(cfg.get('default_scope', 'mmaction'))
        test_pipeline_cfg = cfg.test_pipeline
        test_pipeline = Compose(test_pipeline_cfg)

    input_flag = None
    if isinstance(video, dict):
        input_flag = 'dict'
    elif isinstance(video, str) and osp.exists(video):
        if video.endswith('.npy'):
            input_flag = 'audio'
        else:
            input_flag = 'video'
    else:
        raise RuntimeError(f'The type of argument `video` is not supported: '
                           f'{type(video)}')

    if input_flag == 'dict':
        data = video
    if input_flag == 'video':
        data = dict(filename=video, label=-1, start_index=0, modality='RGB')
    if input_flag == 'audio':
        data = dict(
            audio_path=video,
            total_frames=len(np.load(video)),
            start_index=0,
            label=-1)

    data = test_pipeline(data)
    data = pseudo_collate([data])

    # Forward the model
    with torch.no_grad():
        result = model.test_step(data)[0]

    return result, model.data_preprocessor(data, False)


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file/url')
    parser.add_argument('video', help='video file/url or rawframes directory')
    parser.add_argument('label', help='label file')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--device', type=str, default='cpu', help='CPU/CUDA device option')
    parser.add_argument(
        '--fps',
        default=30,
        type=int,
        help='specify fps value of the output video when using rawframes to '
        'generate file')
    parser.add_argument(
        '--font-scale',
        default=None,
        type=float,
        help='font scale of the text in output video')
    parser.add_argument(
        '--font-color',
        default='white',
        help='font color of the text in output video')
    parser.add_argument(
        '--target-resolution',
        nargs=2,
        default=None,
        type=int,
        help='Target resolution (w, h) for resizing the frames when using a '
        'video as input. If either dimension is set to -1, the frames are '
        'resized by keeping the existing aspect ratio')
    parser.add_argument('--out-filename', default=None, help='output filename')
    args = parser.parse_args()
    return args


def find_ndarray_differences(arr1: np.ndarray, arr2: np.ndarray) -> list:
    """
    计算两个 NumPy 数组 (ndarray) 之间的差值，并返回差值不为零的元素的坐标。
    这两个数组的维度应为 (N, T, C, H, W)。

    Args:
        arr1 (np.ndarray): 第一个数组，形状应为 (N, T, C, H, W)。
        arr2 (np.ndarray): 第二个数组，形状应为 (N, T, C, H, W)。

    Returns:
        list: 一个包含元组的列表，每个元组代表一个差异元素的坐标 (N, T, C, H, W)。
              如果两个数组完全相同，则返回空列表。

    Raises:
        ValueError: 如果输入数组的形状不同，或者它们不是预期的 5 维数组。
    """
    if arr1.shape != arr2.shape:
        raise ValueError(
            f"输入数组的形状必须相同。arr1 的形状: {arr1.shape}, arr2 的形状: {arr2.shape}")

    if arr1.ndim != 5:
        # 尽管形状检查可能已部分覆盖此情况，但明确检查维度确保符合 (N,T,C,H,W) 的结构。
        raise ValueError(f"输入数组必须是 5 维的 (N,T,C,H,W)。实际维度: {arr1.ndim}D。")

    # 计算逐元素的差值
    difference = arr1 - arr2

    # 找到差值不等于零的元素的索引
    # np.where 返回一个元组，其中每个元素是一个数组，对应各个维度的索引
    mismatched_indices = np.where(difference != 0)

    if not mismatched_indices[0].size:
        # 如果第一个索引数组为空，表示没有差异
        return []  # 数组相同
    else:
        # 将索引元组转换为坐标元组的列表
        # 例如: mismatched_indices = (array_N, array_T, array_C, array_H, array_W)
        # zip(*mismatched_indices) 会产生 [(n1,t1,c1,h1,w1), (n2,t2,c2,h2,w2), ...]
        coordinates = list(zip(*mismatched_indices))
        return coordinates


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Build the recognizer from a config file and checkpoint file/url
    model = init_recognizer(cfg, args.checkpoint, device=args.device)
    pred_result, data = inference_recognizer(model, args.video)

    model_input = to_numpy(data['inputs'])
    save_name = "/home/tl/work/mmdeploy/mmdeploy_models/mmaction/tsm/ort/end2end.onnx"

    my_input = preprocess_frames_for_onnx(
        "/home/tl/data/datasets/video/vidoe_recognition_data/1/")

    diff = find_ndarray_differences(model_input, my_input)
    if diff:
        print(f"找到 {len(diff)} 处差异。坐标 (N,T,C,H,W) 及其对应的值如下:")
        for coord in diff:
            val1 = diff[coord]
            val2 = diff[coord]
            print(
                f"  坐标: {coord}, arr1 中的值: {val1}, arr2 中的值: {val2}, 差值: {val1 - val2}")
    else:
        print("未找到差异，但预期应有差异。")

    ort_session = onnxruntime.InferenceSession(save_name)

    ort_inputs = {'input': model_input}

    input_txt_path = "input_data.txt"
    ort_inputs['input'].reshape(-1).tofile(input_txt_path,
                                           sep='\n', format='%f')

    ort_outs = ort_session.run(None, ort_inputs)
    print("onnx outputs:")
    print(ort_outs)


if __name__ == "__main__":
    main()
