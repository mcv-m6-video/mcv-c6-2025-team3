import numpy as np
from utils import split_video_25_75

def compute_mean_and_std(frames):
    """Compute mean and variance of pixels in the first 25% of the 
    test sequence to model background

    Parameters
        frames_25 : np.ndarray([n_frames, 1080, 1920, 1])

    Returns
        mean : np.ndarray([1080, 1920])
        std : np.ndarray([1080, 1920])"""

    # Stack frames into a 3D array: (num_frames, height, width)
    frames_stack = np.stack(frames, axis=0)

    print("Computing mean and std...")
    mean = np.mean(frames_stack, axis=0)
    std = np.std(frames_stack, axis=0)

    return mean, std

def segment_foreground(frames, mean, std, alpha):
    """Return the estimation of the foreground using the reting 75% of the frames

     Returns
        foreground_background : np.ndarray([n_frames, 1080, 1920, 3], dtype=bool)
    """
    print("Segmenting foreground...")
    #implementar pseudocodigo slide 22: week1 instructions
    foreground = np.abs((frames-mean) >= alpha * (std + 2))
    return foreground.astype(bool)

def segment_foreground_chunks(frames, mean, std, alpha):
    n_chunks = 5
    chunks = np.array_split(frames, n_chunks)
    foreground_results = []

    for chunk in chunks:
        foreground = np.abs((chunk-mean) >= alpha * (std + 2))
        foreground_results.append(foreground)

    return np.concatenate(foreground_results, axis=0).astype(bool)

# TASK 1.1: Gaussian modeling
def gaussian_modeling(color_frames, gray_frames, alpha):

    # Separate video in first 25% and second 75%
    color_frames_25, color_frames_75 = split_video_25_75(color_frames)
    gray_frames_25, gray_frames_75 = split_video_25_75(gray_frames)

    # Compute mean and variance of pixels in the first 25% frames
    mean, std = compute_mean_and_std(gray_frames_25)

    # Segment foreground
    foreground_segmented = segment_foreground(gray_frames_75, mean, std, alpha)

    return foreground_segmented, color_frames_75, len(color_frames_25)





