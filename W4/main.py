import argparse
from pathlib import Path
from multitracking import object_tracking_in_all_sequence, evaluate, convert2gps, camera_multitracking


# CHANGE PATHS ADAPTING TO YOUR ABSOLUTE PATH:
track_eval_path = f'/home/danielpardo/c6/TrackEval'

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Week 4')
    parser.add_argument('--task', type=int, default=1,  help=(
        "Introduce the task number you want to execute:\n"
        "  1.Multi-camera tracking\n"
        "  99.Compute object tracking in all cameras of the sequence\n"
    ))
    args = parser.parse_args()

    args.task = 99

    # python main.py --task 1
    if args.task == 1:
        print("Task 1: Multi-camera tracking...")
        # TODO: Implement the multi-camera tracking in multi_camera_tracking.py
    

    elif args.task == 99:
        print("Compute object tracking in all cameras of the sequence...")
        output_folder = Path(f'/home/danielpardo/c6/W4/results_ok')
        output_folder.mkdir(exist_ok=True)

        dataset_path = f'/home/danielpardo/c6/aic19-track1-mtmc-train/train/S04/'

        #object_tracking_in_all_sequence(dataset_path, output_folder, track_eval_path)

        #evaluate(dataset_path, output_folder, track_eval_path)
        #convert2gps(dataset_path, output_folder)
        camera_multitracking(dataset_path, output_folder, track_eval_path)
        
