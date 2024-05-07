import cv2
import numpy as np
from random import randint
import torch
from torchvision.transforms import functional as F


# TODO: convert to input arguments
WINDOW_SIZE = 150
PATCHES_PER_FRAME = 100
FLOW_THRESHOLD_WRT_MAX = 0.5 # needs to be >0 and <1

def get_random_point(w, h):
        return randint(0, h - WINDOW_SIZE), randint(0, w - WINDOW_SIZE)


def select_patches_from_frames(rgb_file_frames, depth_file_frames, prev_idx, gt_idx, next_idx, datapoints):
    """
    This function selects patches from frames based on optical flow. It calculates the optical flow between 
    the previous and next frames, and selects patches where the flow is above a certain threshold. 
    The patches are then added to the provided datapoints list.
    """
    p =  rgb_file_frames[prev_idx]
    g =  rgb_file_frames[gt_idx]
    n =  rgb_file_frames[next_idx]
    frame_samples = []
    frame_size = p.shape
    for _ in range(PATCHES_PER_FRAME):
        py, px = get_random_point(frame_size[0], frame_size[1])
        prev = p[px: px + WINDOW_SIZE, py: py+WINDOW_SIZE]
        prev_d = depth_file_frames[prev_idx][px: px + WINDOW_SIZE, py: py+WINDOW_SIZE]

        gt = g[px: px + WINDOW_SIZE, py: py+WINDOW_SIZE]
        gt_d = depth_file_frames[gt_idx][px: px + WINDOW_SIZE, py: py+WINDOW_SIZE]

        next = n[px: px + WINDOW_SIZE, py: py+WINDOW_SIZE]
        next_d = depth_file_frames[next_idx][px: px + WINDOW_SIZE, py: py+WINDOW_SIZE]

        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY), cv2.cvtColor(next, cv2.COLOR_BGR2GRAY), None, 0.5, 2, WINDOW_SIZE, 3, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        total_flow = np.average(np.linalg.norm(flow, ord=2, axis=2))

        # Convert frames to tensors
        prev_ = torch.vstack((F.to_tensor(prev), F.to_tensor(prev_d)))
        gt_ = torch.vstack((F.to_tensor(gt), F.to_tensor(gt_d)))
        next_ = torch.vstack((F.to_tensor(next), F.to_tensor(next_d)))

        frame_samples.append((total_flow, prev_, gt_, next_))

    # Sort samples by flow and select patches with flow above threshold
    frame_samples.sort(key= lambda x: x[0], reverse=True)
    max_flow = frame_samples[0][0]
    min_flow = frame_samples[-1][0]
    flow_threshold = min_flow + FLOW_THRESHOLD_WRT_MAX*(max_flow-min_flow)
    
    idx = 0
    while idx < len(frame_samples) and frame_samples[idx][0] >= flow_threshold:
        datapoints.append(tuple((frame_samples[idx][1:])))
        idx += 1
    datapoints.append(tuple((frame_samples[-1][1:])))

    del frame_samples