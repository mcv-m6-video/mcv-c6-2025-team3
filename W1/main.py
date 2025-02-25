import argparse
import pickle
import gc

from utils import read_annonations, save_foreground
from task_1 import gaussian_modeling, compute_bbox, bbox2Coco, evaluate
from task_2 import adaptative_modelling
from task_3 import state_of_the_art_background_estimation

# CHANGE PATHS ADAPTING TO YOUR ABSOLUTE PATH:
video_path = r'/Users/andrea.sanchez/Desktop/AICity_data/train/S03/c010/vdo.avi'
annotations_path =  r'/Users/andrea.sanchez/Desktop/ai_challenge_s03_c010-full_annotation.xml'

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Week 1')
    parser.add_argument('--task', type=int, default=1, help='Task number')

    args = parser.parse_args()

    # python main.py --task 1
    if args.task == 1:
        print("Task 1.1: Gaussian modeling")
        bboxes_gt = read_annonations(annotations_path)
        
        for alpha in  [1,4,6]:
            print(f'Alpha: {alpha}')
            foreground_segmented, color_frames_75, first_frame = gaussian_modeling(video_path, alpha)

            with open(f'output_task_1/data_{alpha}.pkl', 'wb') as f:
                pickle.dump({'foreground_segmented': foreground_segmented, 'color_frames_75': color_frames_75, 'first_frame': first_frame}, f)
            
            save_foreground(foreground_segmented, color_frames_75, alpha)

            """
            with open(f'data_{alpha}.pkl', 'rb') as f:
                data = pickle.load(f)

            foreground_segmented = data['foreground_segmented']
            color_frames_75 = data['color_frames_75']
            first_frame = data['first_frame']
            """
            
            bboxes_predict = compute_bbox(foreground_segmented, color_frames_75, alpha, first_frame+1)

            """
            with open(f'bbox_task1_{str(alpha)}.pkl', 'rb') as f:
                data = pickle.load(f)

            bboxes_predict = data['bboxes']
            """
            
            #create coco.json
            bbox2Coco(bboxes_gt, alpha, 'gt')
            bbox2Coco(bboxes_predict, alpha, 'predict', f'output_task_1/coco_gt_{alpha}.json')
            
            result_global, result, result_auto = evaluate(color_frames_75, first_frame+1, alpha, 0.50, f'output_task_1/coco_gt_{alpha}.json', f'output_task_1/coco_predict_{alpha}.json')

            print(f'Alpha: {alpha}, Global: {result_global:.4f},  byFrame: {result:.4f}, byFrame auto: {result_auto:.4f}')

            del foreground_segmented, color_frames_75, first_frame, bboxes_predict
            gc.collect()

    elif args.task == 2:
        print("Task 2.1: Adaptative modeling")
        segmented_frames, color_frames_75 = adaptative_modelling(video_path, annotations_path, alpha=11, p=0.01)
        
    elif args.task == 3:
        print("Task 3.1: SOTA evaluation")
        state_of_the_art_background_estimation(video_path, annotations_path, technique='MOG2')