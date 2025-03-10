import argparse
from pathlib import Path
import os
import numpy as np
import cv2
from task_1_1 import optical_flow_off_the_shelf, generate_optical_flow_legend,compute_msen_pepn, read_flow_gt, flow_to_color
from task_1_2 import create_folder_structure, annonations2mot, evaluate_tracking, generate_optical_flow_video_gif, KalmanFilterWithOpticalFlow

# CHANGE PATHS ADAPTING TO YOUR ABSOLUTE PATH:
video_path = f'/home/danielpardo/c6/AICity_data/train/S03/c010/vdo.avi'
annotations_path =  f'/home/danielpardo/c6/ai_challenge_s03_c010-full_annotation.xml'
track_eval_path = f'/home/danielpardo/c6/TrackEval'


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Week 3')
    parser.add_argument('--task', type=int, default=1,  help=(
        "Introduce the task number you want to execute:\n"
        "  1.Optical flow off-the-shelf\n"
        "  2.Improve tracking with optical flow\n"
        "  3.Evaluate best tracking algorithm in SEQ01\n"
        "  4.Evaluate best tracking algorithm in SEQ03\n"
    ))
    args = parser.parse_args()

    args.task = 2

    # python main.py --task 1
    if args.task == 1:
        print("Task 1.1: Optical flow off-the-shelf")
        output_folder = Path('/home/danielpardo/c6/W3/output_task_1')
        output_folder.mkdir(exist_ok=True)

        subset = 'image_0' #'colored_0'
        kitti_path = f'/home/danielpardo/c6/data_stereo_flow/training/{subset}/'
        
        images = os.listdir(kitti_path)
        sequence_45_images = sorted([image for image in images if '000045' in image])

        output_file = f"{output_folder}/results_{subset}.txt"

        with open(output_file, "w") as f:
            for gt_option in ['noc', 'occ']:
                flow_gt, valid_gt = read_flow_gt(f"/home/danielpardo/c6/data_stereo_flow/training/flow_{gt_option}/000045_10.png")
                gt_img = flow_to_color(flow_gt[..., 0], flow_gt[..., 1])
                cv2.imwrite(f'{output_folder}/optical_flow_GT_{gt_option}_{subset}.png', gt_img)
                for option in ['maskflownet', 'raft', 'craft', 'memflow' ,'pyflow', 'farneback']:
                    
                    flow, inf_time = optical_flow_off_the_shelf(output_folder, kitti_path + sequence_45_images[0], kitti_path + sequence_45_images[1], option, subset)
                    
                    # metrics 
                    msen, pepn, motion_vectors = compute_msen_pepn(flow, flow_gt, valid_gt)

                    cv2.imwrite(f'{output_folder}/error_map_{option}_{gt_option}_{subset}.png', 
                                cv2.applyColorMap(cv2.normalize(motion_vectors, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET))

                    f.write(f"GT Option: {gt_option} | Method: {option} | MSEN: {msen:.4f} | PEPN: {pepn:.2f}% | Time: {inf_time:.4f}\n")
                    f.write("-" * 90 + "\n")


    elif args.task == 2:
        print("Task 1.2: Improve tracking with optical flow")
        output_folder = Path('/home/danielpardo/c6/W3/output_task_2')
        output_folder.mkdir(exist_ok=True)

        output_file = f"{output_folder}/results_tracking.txt"
        
        with open(output_file, "w") as f:
            for option in [False]:
                for max_age in [5, 9, 15]:
                    for min_hits in [2, 3]:
                        for iou_threshold in [0.2, 0.3, 0.5]:
                            kalman_filter = KalmanFilterWithOpticalFlow(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold, optical_flow=option)
                            kalman_filter.execute(video_path, output_folder, track_eval_path)
                            hota, idf1 = evaluate_tracking(track_eval_path)
                            f.write(f"HOTA: {hota:.4f} | IDF1: {idf1:.4f} | Max age: {max_age} | Min hits: {min_hits} | IoU: {iou_threshold:.2f}\n")
                            f.write("-" * 90 + "\n")


    elif args.task == 3:
        print("Task 2.1: Evaluate best tracking algorithm in SEQ01")

    elif args.task == 4:
        print("Task 2.2: Evaluate best tracking algorithm in SEQ03")
        
    elif args.task == 99:
        output_folder = Path('/home/danielpardo/c6/W3/output_task_1')
        output_folder.mkdir(exist_ok=True)

        #generate_optical_flow_legend(output_folder)
        #create_folder_structure(track_eval_path)
        #annonations2mot(annotations_path, f'{track_eval_path}/data/gt/mot_challenge/custom-train/s03/gt/gt.txt')
        #generate_optical_flow_video_gif(output_folder, video_path)

    else:
        print('Task not implemented')
        exit(1)
