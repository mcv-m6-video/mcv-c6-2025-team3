import argparse
from pathlib import Path

import os
from utils_1 import read_annonations, trim_gif, evaluate, video2frames, get_dataloader, data2yolo
from task_1 import detect_cars_yolov8n, detect_cars_fasterrcnn, finetuneYolo, finetuneFasterrcnn
from task_2_1 import object_tracking_by_overlap
from task_2_2 import object_tracking_by_kalman_filter
from utils import create_gif, read_annonations, trim_gif
from task_1_1 import detect_cars_yolov8n

# CHANGE PATHS ADAPTING TO YOUR ABSOLUTE PATH:
video_path = r'/Users/papallusqueti/Downloads/AICity_data/train/S03/c010/vdo.avi'
annotations_path =  r'/Users/papallusqueti/Downloads/ai_challenge_s03_c010-full_annotation.xml'
image_folder = r'/home/danielpardo/c6/frames'
yolo_folder = r'/home/danielpardo/c6/yolo_data/'

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Week 2')
    parser.add_argument('--task', type=int, default=1,  help=(
        "Introduce the task number you want to execute:\n"
        "  1.Object detection off-the-shelf\n"
        "  2.Fine-tune to our data\n"
        "  3.K-Fold Cross Validation\n"
        "  4.Object tracking by overlap"
        "  5.Object tracking by Kalman filter"
        "  6.Comparing metrics across IoU thresholds"
        "  99.Data preparation"
    ))

    args = parser.parse_args()

    bboxes_gt = read_annonations(annotations_path)

    # python main.py --task 1
    if args.task == 1:
        print("Task 1.1: Object detection off-the-shelf...")

        output_folder = Path('/home/danielpardo/c6/W2/output_task_1')
        output_folder.mkdir(exist_ok=True)

        gt_annotations, frames_gt = read_annonations(annotations_path)
        image_paths = sorted([os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(".jpg")])
        conf_th = 0.3
        byCar = True
        option = 'yolov8n'

        if option == 'yolov8n':
            predictions = detect_cars_yolov8n(video_path, gt_annotations, frames_gt, conf_th, output_folder, byCar=byCar, gif=True)

        if option == 'fasterrcnn':
            predictions = detect_cars_fasterrcnn(video_path, gt_annotations, frames_gt, conf_th, output_folder, byCar=byCar, gif=True)

        evaluate(gt_annotations, predictions, f'{output_folder}/results_{option}_{conf_th}_{byCar}.txt')

        # Cut the gif to 10 seconds for power point
        cut_gif = f'{output_folder}/{option}_{conf_th}_{byCar}_predict.gif'
        trim_gif(cut_gif, f'{output_folder}/{option}_{conf_th}_{byCar}_predict_trimmed.gif', start_time=50, end_time=60)
        cut_gif = f'{output_folder}/{option}_{conf_th}_{byCar}_predict_gt.gif'
        trim_gif(cut_gif, f'{output_folder}/{option}_{conf_th}_{byCar}_predict_gt_trimmed.gif', start_time=50, end_time=60)

    elif args.task == 2:
        print("Task 1.2: Fine-tune to our data...")

        output_folder = Path('/home/danielpardo/c6/W2/output_task_2')
        output_folder.mkdir(exist_ok=True)

        gt_annotations, frames_gt = read_annonations(annotations_path)
        image_paths = sorted([os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(".jpg")])
        
        option = 'fasterrcnn'
        strategy = 'b'
        batch_size = 8
        conf_th = 0.3

        for option in ['fasterrcnn', 'yolov8n']:
            for strategy in ['a', 'b', 'c']:
                output_option = Path(f'{output_folder}/{option}')
                output_option.mkdir(exist_ok=True)

                if option == 'yolov8n':
                    epochs = 20
                    finetuneYolo(yolo_folder, output_option, strategy, batch_size, epochs)
                elif option == 'fasterrcnn':
                    epochs = 10
                    dataloaders = get_dataloader(image_paths, gt_annotations, batch_size, strategy)
                    finetuneFasterrcnn(dataloaders, output_option, strategy, epochs, conf_th)

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
        output_folder = Path('output_task_5')
        output_folder.mkdir(exist_ok=True)
        object_tracking_by_kalman_filter(video_path, output_folder)

        # Cut the gif to 10 seconds for power point
        # cut_gif = output_folder / 'kalman_tracking.gif'
        # trim_gif(cut_gif, output_folder / 'kalman_trimmed_15_25.gif', start_time=15, end_time=25)
    
    elif args.task == 6:
        print("Task 2.3: Comparing metrics across IoU thresholds...")
        from task_2_3 import compare_metrics_across_thresholds
        # Call the function with the base output folder that contains your min_iou_* folders.
        compare_metrics_across_thresholds(Path('output_task_6'))

    elif args.task == 99:
        print("Data preparation...")

        print("Video to frames...")
        video2frames(video_path)
        
        gt_annotations, frames_gt = read_annonations(annotations_path)
        image_paths = sorted([os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(".jpg")])

        print("Data to yolo...")
        data2yolo(image_paths, gt_annotations, yolo_folder)

    else:
        print('Task not implemented')
        exit(1)
