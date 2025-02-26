import numpy as np
import gc

from utils import read_video, split_video_25_75, save_foreground, compute_bbox, bbox2Coco, evaluate

# TASK 2.1: Adaptative modeling
def variable_background_modeling(gray_frames_25):
    print('Modeling background...')
    number_frames = gray_frames_25.shape[0]
    height, width = gray_frames_25.shape[1], gray_frames_25.shape[2]

    # Initialize the mean and variance with the first frame
    mean = np.zeros((height, width))
    variance = np.zeros((height, width))

    # mean
    for i in range(number_frames):
        frame = gray_frames_25[i].astype(np.float32)
        mean += (frame - mean) / (i + 1)

    # variance
    for i in range(number_frames):
        frame = gray_frames_25[i].astype(np.float32)
        variance += (frame - mean) ** 2

    # Compute the standard deviation from the variance
    variance /= (number_frames-1) #-1 bessel correction
    std = np.sqrt(variance)

    return mean, variance, std

# TASK 2.1: Adaptative modeling
def adaptative_modelling(video_path, alpha, p):
    # Read video to get frames from it
    color_frames, gray_frames = read_video(video_path)

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
        mask = np.abs(frame - mean) >= alpha * (std+2)
        fg_mask = mask.astype(bool)
        
        # Update background model for background pixels only
        bg_pixels = ~mask
        mean[bg_pixels] = p * frame[bg_pixels] + (1 - p) * mean[bg_pixels]
        variance[bg_pixels] = p * (frame[bg_pixels] - mean[bg_pixels]) ** 2 + (1 - p) * variance[bg_pixels]
        std = np.sqrt(variance)
        
        # Add the segmented frame to the list
        segmented_frames.append(fg_mask)
    return np.array(segmented_frames), color_frames_75, len(color_frames_25)


def process(video_path, bboxes_gt, alpha, rho, output_folder):
    segmented_frames, color_frames_75, first_frame = adaptative_modelling(video_path, alpha, rho)

    save_foreground(segmented_frames, color_frames_75, alpha, output_folder, rho)

    bboxes_predict = compute_bbox(segmented_frames, color_frames_75, alpha, first_frame+1, output_folder, rho)

    bbox2Coco(bboxes_gt, f'{alpha}_{rho}', 'gt', output_folder)
    bbox2Coco(bboxes_predict, f'{alpha}_{rho}', 'predict', output_folder, f'{output_folder}/coco_gt_{alpha}_{rho}.json')
    
    result_global, result, result_auto = evaluate(color_frames_75, first_frame+1, alpha, 0.50, f'{output_folder}/coco_gt_{alpha}_{rho}.json',  f'{output_folder}/coco_predict_{alpha}_{rho}.json', output_folder, rho)
    
    print(f'Alpha: {alpha}, rho: {rho}, Global: {result_global:.4f},  byFrame: {result:.4f}, byFrame auto: {result_auto:.4f}')

    del segmented_frames, color_frames_75, first_frame, bboxes_predict
    gc.collect()    

    return result_global


def find_alpha(video_path, bboxes_gt, output_folder):
    rho = 0.05
    best_alpha = 0
    best_result = 0
    for alpha in [2, 2.5, 3, 5, 7, 9, 11]:
        result = process(video_path, bboxes_gt, alpha, rho, output_folder)
        if result > best_result:
            best_result = result
            best_alpha = alpha

    return best_alpha


def find_rho(video_path, bboxes_gt, alpha, output_folder):

    best_rho = 0
    best_result = 0

    for rho in [0.01, 0.03, 0.07, 0.1, 0.3]:
        result = process(video_path, bboxes_gt, alpha, rho, output_folder)
        if result > best_result:
            best_result = result
            best_rho = rho
            
    return best_rho