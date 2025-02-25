import argparse

from utils import visualize_foreground
from task_1 import gaussian_modeling
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
        foreground_segmented, color_frames_75 = gaussian_modeling(video_path, annotations_path, alpha=11)
        visualize_foreground(foreground_segmented, color_frames_75)

    elif args.task == 2:
        print("Task 2.1: Adaptative modeling")
        segmented_frames, color_frames_75 = adaptative_modelling(video_path, annotations_path, alpha=11, p=0.01)
        
    elif args.task == 3:
        print("Task 3.1: SOTA evaluation")
        state_of_the_art_background_estimation(video_path, annotations_path, technique='MOG2')