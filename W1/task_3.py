import cv2
import numpy as np
from utils import split_video_25_75, frames2gif, bbox_dict, bbox2Coco, evaluate
import gc

def state_of_the_art_background_estimation(color_frames, gray_frames, bboxes_gt, output_folder, technique):

    # Create background substractor model depending on the technique chosen
    # if technique == 'MOG':
    #     bg_subtractor = cv2.createBackgroundSubtractorMOG() #deprectaed

    if technique == 'MOG2':
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=50,varThreshold=25)
    elif technique == 'LSBP':
        bg_subtractor = cv2.bgsegm.createBackgroundSubtractorLSBP(LSBPRadius=10, Tlower=2.0, Tupper=32.0, Tinc=1.0, Tdec=0.05)
    elif technique == 'LOBSTER':
        pass
    elif technique == 'CNT':
        bg_subtractor = cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=15, useHistory=True, maxPixelStability=50, isParallel=True)
    elif technique == 'GSOC':    
        bg_subtractor = cv2.bgsegm.createBackgroundSubtractorGSOC()
    elif technique == 'KNN':
        bg_subtractor = cv2.createBackgroundSubtractorKNN(history=50, dist2Threshold=300)
    else:
        print('Technique not implemented')
        return

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

    first_frame = len(color_frames_25)

    output_path = f'{output_folder}/foreground_task3_{technique}.avi'
    fps = 30 
    frame_size = (color_frames_75.shape[2], color_frames_75.shape[1])
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)
    frames = []
    for i in range(foreground_estimation.shape[0]):
        frame = color_frames_75[i].copy()

        frame[foreground_estimation[i]] = [0, 0, 255]

        out.write(frame)
        frames.append(frame)
    out.release()
    frames2gif(frames, f'{output_folder}/foreground_task3_{technique}.gif')
    
    bboxes_predict = bbox_dict(foreground_estimation, first_frame+1, f'{output_folder}/bbox_task3_{technique}.pkl')

    bbox2Coco(bboxes_gt, technique, 'gt', output_folder)
    bbox2Coco(bboxes_predict, technique, 'predict', output_folder, f'{output_folder}/coco_gt_{technique}.json')
    
    result_global, result, result_auto = evaluate(color_frames_75, first_frame+1, technique, 0.50, f'{output_folder}/coco_gt_{technique}.json',  f'{output_folder}/coco_predict_{technique}.json', output_folder)

    print(f'Technique: {technique}, Global: {result_global:.4f},  byFrame: {result:.4f}, byFrame auto: {result_auto:.4f}')
    
    del color_frames, color_frames_25, color_frames_75 
    del gray_frames, gray_frames_25, gray_frames_75 
    del foreground_estimation, first_frame, bboxes_predict
    gc.collect()