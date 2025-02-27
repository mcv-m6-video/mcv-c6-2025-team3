import argparse
import gc
from pathlib import Path
import cv2
import numpy as np
from utils import cut_gif, read_annonations, save_foreground, compute_bbox, bbox2Coco, evaluate, read_video
from task_1 import gaussian_modeling
from task_2 import find_alpha, find_rho
from task_3 import state_of_the_art_background_estimation

# CHANGE PATHS ADAPTING TO YOUR ABSOLUTE PATH:
video_path = r'/Users/andrea.sanchez/Desktop/AICity_data/train/S03/c010/vdo.avi'
annotations_path =  r'/Users/andrea.sanchez/Desktop/ai_challenge_s03_c010-full_annotation.xml'

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Week 1')
    parser.add_argument('--task', type=int, default=1, help='Task number')

    args = parser.parse_args()

    color_frames, gray_frames = read_video(video_path)

    bboxes_gt = read_annonations(annotations_path)

    # python main.py --task 1
    if args.task == 1:
        print("Task 1.1: Gaussian modeling")
        output_folder = Path('output_task_1')
        output_folder.mkdir(exist_ok=True)


        for alpha in [2,4,6,9,10,11,12,13]:
            print(f'Alpha: {alpha}')
            foreground_segmented, color_frames_75, first_frame = gaussian_modeling(color_frames.copy(), gray_frames.copy(), alpha)

            save_foreground(foreground_segmented, color_frames_75, alpha, output_folder)

            bboxes_predict = compute_bbox(foreground_segmented, color_frames_75, alpha, first_frame+1, output_folder)

            #create coco.json
            bbox2Coco(bboxes_gt, alpha, 'gt', output_folder)
            bbox2Coco(bboxes_predict, alpha, 'predict', output_folder, f'{output_folder}/coco_gt_{alpha}.json')
            
            result_global, result, result_auto = evaluate(color_frames_75, first_frame+1, alpha, 0.50, f'{output_folder}/coco_gt_{alpha}.json',  f'{output_folder}/coco_predict_{alpha}.json', output_folder)

            print(f'Alpha: {alpha}, Global: {result_global:.4f},  byFrame: {result:.4f}, byFrame auto: {result_auto:.4f}')

            del foreground_segmented, color_frames_75, first_frame, bboxes_predict
            gc.collect()

    elif args.task == 2:
        print("Task 2.1: Adaptative modeling")
        output_folder = Path('output_task_2')
        output_folder.mkdir(exist_ok=True)

        alpha = find_alpha(color_frames.copy(), gray_frames.copy(), bboxes_gt, output_folder)
        rho = find_rho(color_frames.copy(), gray_frames.copy(), bboxes_gt, alpha,output_folder)

    elif args.task == 3:
        output_folder = Path('output_task_3')
        output_folder.mkdir(exist_ok=True)
        print("Task 3.1: SOTA evaluation")
        for technique in ['MOG','MOG2','LSBP', 'CNT', 'GSOC', 'KNN']:
            try:
                state_of_the_art_background_estimation(color_frames.copy(), gray_frames.copy(), bboxes_gt, output_folder, technique)
            except:
                print("ERROR", technique)
    elif args.task == 4:
        print("cutting gifs...")
        # Example usage:
        cut_gif("/Users/andrea.sanchez/Desktop/mcv-c6-2025-team3/W1/output_task_3/morph_task_MOG2.gif", "gif/morph_task_MOG2.gif", 20) 
        cut_gif("/Users/andrea.sanchez/Desktop/mcv-c6-2025-team3/W1/output_task_3/pre_morph_task_MOG2.gif", "gif/pre_morph_task_MOG2.gif", 20)
        cut_gif("/Users/andrea.sanchez/Desktop/mcv-c6-2025-team3/W1/output_task_3/morph_task_LSBP.gif", "gif/morph_task_LSBP.gif", 20) 
        cut_gif("/Users/andrea.sanchez/Desktop/mcv-c6-2025-team3/W1/output_task_3/pre_morph_task_LSBP.gif", "gif/pre_morph_task_LSBP.gif", 20)
        cut_gif("/Users/andrea.sanchez/Desktop/mcv-c6-2025-team3/W1/output_task_3/morph_task_CNT.gif", "gif/morph_task_CNT.gif", 20)
        cut_gif("/Users/andrea.sanchez/Desktop/mcv-c6-2025-team3/W1/output_task_3/pre_morph_task_CNT.gif", "gif/pre_morph_task_CNT.gif", 20)
        cut_gif("/Users/andrea.sanchez/Desktop/mcv-c6-2025-team3/W1/output_task_3/morph_task_GSOC.gif", "gif/morph_task_GSOC.gif", 20)
        cut_gif("/Users/andrea.sanchez/Desktop/mcv-c6-2025-team3/W1/output_task_3/pre_morph_task_GSOC.gif", "gif/pre_morph_task_GSOC.gif", 20)
        cut_gif("/Users/andrea.sanchez/Desktop/mcv-c6-2025-team3/W1/output_task_3/morph_task_KNN.gif", "gif/morph_task_KNN.gif", 20)
        cut_gif("/Users/andrea.sanchez/Desktop/mcv-c6-2025-team3/W1/output_task_3/pre_morph_task_KNN.gif", "gif/pre_morph_task_KNN.gif", 20)

    else:
        print('Task not implemented')
        exit(1)
