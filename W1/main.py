import argparse
import gc
from pathlib import Path

from utils import read_annonations, save_foreground, compute_bbox, bbox2Coco, evaluate
from task_1 import gaussian_modeling
from task_2 import find_alpha, find_rho
from task_3 import state_of_the_art_background_estimation

# CHANGE PATHS ADAPTING TO YOUR ABSOLUTE PATH:
video_path = r'./AICity_data/train/S03/c010/vdo.avi'
annotations_path =  r'./ai_challenge_s03_c010-full_annotation.xml'

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Week 1')
    parser.add_argument('--task', type=int, default=1, help='Task number')

    args = parser.parse_args()

    args.task = 2

    # python main.py --task 1
    if args.task == 1:
        print("Task 1.1: Gaussian modeling")
        output_folder = Path('output_task_1')
        output_folder.mkdir(exist_ok=True)

        bboxes_gt = read_annonations(annotations_path)

        for alpha in [2,4,6,9,10,11,12,13]:
            print(f'Alpha: {alpha}')
            foreground_segmented, color_frames_75, first_frame = gaussian_modeling(video_path, alpha)

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
        bboxes_gt = read_annonations(annotations_path)

        alpha = find_alpha(video_path, bboxes_gt, output_folder)
        rho = find_rho(video_path, bboxes_gt, alpha,output_folder)

    elif args.task == 3:
        output_folder = Path('output_task_3')
        output_folder.mkdir(exist_ok=True)
        print("Task 3.1: SOTA evaluation")
        state_of_the_art_background_estimation(video_path, annotations_path, technique='MOG2')