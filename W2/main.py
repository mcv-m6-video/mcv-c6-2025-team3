import argparse
from pathlib import Path

from utils import read_annonations, trim_gif
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
        print("Task 2: Object tracking...")
        
    else:
        print('Task not implemented')
        exit(1)
