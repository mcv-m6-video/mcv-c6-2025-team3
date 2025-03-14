import argparse
from pathlib import Path


# CHANGE PATHS ADAPTING TO YOUR ABSOLUTE PATH:
video_path = r'/Users/andrea.sanchez/Desktop/AICity_data/train/S03/c010/vdo.avi'
annotations_path =  r'/Users/andrea.sanchez/Desktop/ai_challenge_s03_c010-full_annotation.xml'
track_eval_path = r'/Users/andrea.sanchez/Desktop/TrackEval'


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Week 4')
    parser.add_argument('--task', type=int, default=1,  help=(
        "Introduce the task number you want to execute:\n"
        "  1.Multi-camera tracking\n"
    ))
    args = parser.parse_args()


    # python main.py --task 1
    if args.task == 1:
        print("Task 1: Multi-camera tracking...")
        
