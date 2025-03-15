import argparse
from pathlib import Path
from object_tracking import object_tracking_in_all_sequence


# CHANGE PATHS ADAPTING TO YOUR ABSOLUTE PATH:
dataset_path = r'/Users/andrea.sanchez/Desktop/aic19-track1-mtmc-train/train/S01'


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Week 4')
    parser.add_argument('--task', type=int, default=1,  help=(
        "Introduce the task number you want to execute:\n"
        "  1.Multi-camera tracking\n"
        "  99.Compute object tracking in all cameras of the sequence\n"
    ))
    args = parser.parse_args()

    # python main.py --task 1
    if args.task == 1:
        print("Task 1: Multi-camera tracking...")
        # TODO: Implement the multi-camera tracking in multi_camera_tracking.py
    

    elif args.task == 99:
        print("Compute object tracking in all cameras of the sequence...")
        output_folder = Path('results')
        output_folder.mkdir(exist_ok=True)
        object_tracking_in_all_sequence(dataset_path, output_folder)
        
        
