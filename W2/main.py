import argparse
from pathlib import Path

from task_2_1 import object_tracking_by_overlap
from utils import create_gif, read_annonations, trim_gif
from task_1_1 import detect_cars_yolov8n

# CHANGE PATHS ADAPTING TO YOUR ABSOLUTE PATH:
video_path = r'/Users/andrea.sanchez/Desktop/AICity_data/train/S03/c010/vdo.avi'
annotations_path =  r'/Users/andrea.sanchez/Desktop/ai_challenge_s03_c010-full_annotation.xml'

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Week 2')
    parser.add_argument('--task', type=int, default=1,  help=(
        "Introduce the task number you want to execute:\n"
        "  1.Object detection off-the-shelf\n"
        "  2.Fine-tune to our data\n"
        "  3.K-Fold Cross Validation\n"
        "  4.Object tracking"
    ))

    args = parser.parse_args()

    bboxes_gt = read_annonations(annotations_path)

    # python main.py --task 1
    if args.task == 1:
        output_folder = Path('output_task_1')
        output_folder.mkdir(exist_ok=True)
        print("Task 1.1: Object detection off-the-shelf...")
        predicted_bbxes = detect_cars_yolov8n(video_path, output_folder)

        # Cut the gif to 10 seconds for power point
        cut_gif = output_folder / 'yolov8n_off_shelf_detection.gif'
        trim_gif(cut_gif, output_folder / 'yolov8n_off_shelf_detection_trimmed.gif', start_time=50, end_time=60)

    elif args.task == 2:
        print("Task 1.2: Fine-tune to our data...")

    elif args.task == 3:
        print("Task 1.3: K-Fold Cross Validation...")
        
    elif args.task == 4:
        print("Task 2.1: Object tracking by overlap...")
        output_folder = Path('output_task_4')
        output_folder.mkdir(exist_ok=True)
        object_tracking_by_overlap(video_path, output_folder)

        # Create gifs of tracking detections from different video moments
        # start_frame = 216
        # end_frame = 254
        # create_gif(output_folder/ 'gifs'/ f'yolo_overlap_{start_frame}_{end_frame}.gif', start_frame, end_frame, output_folder)
        
    elif args.task == 5:
        print("Task 2.2: Object tracking by Kalman filter...")
    
    elif args.task == 6:
        print("Task 2.3: Object tracking evaluation...")

    else:
        print('Task not implemented')
        exit(1)
