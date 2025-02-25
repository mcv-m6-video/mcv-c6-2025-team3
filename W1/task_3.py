import cv2
import numpy as np
from utils import read_video, read_annonations, split_video_25_75

def state_of_the_art_background_estimation(video_path, annotations_path, technique):

    # Create background substractor model depending on the technique chosen
    # if technique == 'MOG':
    #     bg_subtractor = cv2.createBackgroundSubtractorMOG() #deprectaed

    if technique == 'MOG2':
        bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    elif technique == 'LSBP':
        bg_subtractor = cv2.createBackgroundSubtractorLSBP()
    elif technique == 'LOBSTER':
        pass
    elif technique == 'GMG':
        bg_subtractor = cv2.bgsegm.createBackgroundSubtractorGMG()
    elif technique == 'CNT':
        bg_subtractor = cv2.bgsegm.createBackgroundSubtractorCNT()
    elif technique == 'GSOC':    
        bg_subtractor = cv2.bgsegm.createBackgroundSubtractorGSOC()
    elif technique == 'KNN':
        bg_subtractor = cv2.createBackgroundSubtractorKNN()
    else:
        print('Technique not implemented')
        return
    
    # Read video to get frames from it
    color_frames, gray_frames = read_video(video_path)

    # Get ground truth annotations
    car_bbxes = read_annonations(annotations_path)

    # Separate video in first 25% and second 75%
    color_frames_25, color_frames_75 = split_video_25_75(color_frames)
    gray_frames_25, gray_frames_75 = split_video_25_75(gray_frames)

    # Train substractor with the first 25% of the frames
    for frame in gray_frames_25:
        bg_subtractor.apply(frame)
    
    # Estimate the foreground in the second 75% of the frames
    foreground_estimation = []
    for frame in gray_frames_75:
        fg_mask = bg_subtractor.apply(frame)

        # Post-process the mask
        if len(fg_mask.shape) == 3:
            fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_BGR2GRAY)
            _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        bool_fg_mask = fg_mask.astype(bool)
        foreground_estimation.append(bool_fg_mask)

    foreground_estimation = np.array(foreground_estimation)

    # Compute metrics