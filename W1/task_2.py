import numpy as np

from utils import read_video, read_annonations, split_video_25_75


# TASK 2.1: Adaptative modeling
def variable_background_modeling(gray_frames_25):
    print('Modeling background...')
    number_frames = gray_frames_25.shape[0]
    height, width = gray_frames_25.shape[1], gray_frames_25.shape[2]

    # Initialize the mean and variance with the first frame
    mean = np.zeros((height, width))
    variance = np.zeros((height, width))

    for i in range(number_frames):
        # Get the frame
        frame = gray_frames_25[i]
        
        # Update the mean
        mean +=  (frame - mean) / (i + 1)
        
        # Update the variance 
        variance += (frame - mean) ** 2 
        
    # Compute the standard deviation from the variance
    std = np.sqrt(variance / (i + 1))

    return mean, variance, std

# TASK 2.1: Adaptative modeling
def adaptative_modelling(video_path, annotations_path, alpha, p):
    # Read video to get frames from it
    color_frames, gray_frames = read_video(video_path)

    # Get ground truth annotations
    car_bbxes = read_annonations(annotations_path)

    # Separate video in first 25% and second 75%
    color_frames_25, color_frames_75 = split_video_25_75(color_frames)
    gray_frames_25, gray_frames_75 = split_video_25_75(gray_frames)

    # Background modeling
    mean, variance, std = variable_background_modeling(gray_frames_25)

    # Segment the frames in the second 75% using the updated background model
    segmented_frames = []
    
    for i in range(gray_frames_75.shape[0]):
        frame = gray_frames_75[i]
        
        # Segmented frame
        mask = np.abs(frame - mean) >= alpha * (std + 2)
        fg_mask = mask.astype(np.uint8) * 255 #0 for bg, 255 for fg
        
        # Update background model for background pixels only
        bg_pixels = ~mask
        mean[bg_pixels] = p * frame[bg_pixels] + (1 - p) * mean[bg_pixels]
        variance[bg_pixels] = p * (frame[bg_pixels] - mean[bg_pixels]) ** 2 + (1 - p) * variance[bg_pixels]
        std = np.sqrt(variance)
        
        # Add the segmented frame to the list
        segmented_frames.append(fg_mask)
    
    # Evaluation and computation of metrics

    
    return segmented_frames, color_frames_75

