import argparse
from pathlib import Path
import os
from utils_1 import read_annonations, trim_gif, evaluate, video2frames, get_dataloader, data2yolo
from task_1 import detect_cars_yolov8n, detect_cars_fasterrcnn, finetuneYolo, finetuneFasterrcnn

# CHANGE PATHS ADAPTING TO YOUR ABSOLUTE PATH:
video_path = r'/home/danielpardo/c6/AICity_data/train/S03/c010/vdo.avi'
annotations_path =  r'/home/danielpardo/c6/ai_challenge_s03_c010-full_annotation.xml'
image_folder = r'/home/danielpardo/c6/frames'
yolo_folder = r'/home/danielpardo/c6/yolo_data/'

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Week 2')
    parser.add_argument('--task', type=int, default=1,  help=(
        "Introduce the task number you want to execute:\n"
        "  1.Object detection off-the-shelf\n"
        "  2.Fine-tune to our data: 3 strategies\n"
        "  99.Data preparation"
    ))

    args = parser.parse_args()

    args.task = 2

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
