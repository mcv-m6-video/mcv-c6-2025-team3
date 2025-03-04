import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import subprocess
import argparse
import imageio
from pathlib import Path
import trackeval

from task_2_1 import Tracker, Detection
from task_1_1 import detect_cars_yolov8n

def convert_gt_xml_to_mot(xml_path, output_path, classes={"car": 2, "bike": 1}):
    """
    Convert ground truth XML to extended MOTChallenge format with class information and save to output_path.
    
    Extended MOT format per line:
      frame, track_id, x, y, width, height, score, class, -1, -1
      
    If the track label is not in the provided mapping, the class field is set to -1.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []
    for track in root.findall('track'):
        track_id = int(track.attrib.get('id'))
        label = track.attrib.get('label')
        if classes is not None and label in classes:
            class_id = classes[label]
        else:
            class_id = -1
        for box in track.findall('box'):
            frame = int(box.attrib.get('frame'))
            xtl = float(box.attrib.get('xtl'))
            ytl = float(box.attrib.get('ytl'))
            xbr = float(box.attrib.get('xbr'))
            ybr = float(box.attrib.get('ybr'))
            width = xbr - xtl
            height = ybr - ytl
            line = f"{frame},{track_id},{xtl:.2f},{ytl:.2f},{width:.2f},{height:.2f},1,{class_id},-1,-1"
            lines.append(line)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Ground truth saved to {output_path}")

def write_tracker_results(results, output_path):
    """
    Write computed tracker results in MOTChallenge format.
    Each line: frame, track_id, x, y, width, height, score, -1, -1, -1
    """
    lines = []
    for res in results:
        frame, track_id, x, y, width, height, score = res
        line = f"{frame},{track_id},{x:.2f},{y:.2f},{width:.2f},{height:.2f},{score:.2f},-1,-1,-1"
        lines.append(line)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Tracker results saved to {output_path}")

def run_evaluation(video_path, gt_xml_path, output_folder):
    """
    Runs the tracking evaluation using TrackEval.
    
    Parameters:
      video_path (str): Path to the video file.
      gt_xml_path (str): Path to the ground truth XML file.
      output_folder (str or Path): Folder where results and visualization are saved.
    """
    output_folder = Path(output_folder).absolute()
    os.makedirs(output_folder, exist_ok=True)

    seq_name = "seq01"
    gt_folder = os.path.join(output_folder, "gt", seq_name)
    tracker_folder = os.path.join(output_folder, "tracker", seq_name)
    os.makedirs(gt_folder, exist_ok=True)
    os.makedirs(tracker_folder, exist_ok=True)

    gt_mot_path = os.path.join(gt_folder, "gt.txt")
    convert_gt_xml_to_mot(gt_xml_path, gt_mot_path, classes={"car": 2, "bike": 1})

    print("Running object detection using YOLOv8n...")
    predictions = detect_cars_yolov8n(video_path, output_folder)
    
    print("Initializing tracker...")
    tracker = Tracker(min_iou=0.1, max_no_detect=10)

    cap = cv2.VideoCapture(video_path)
    computed_results = [] 
    vis_frames = []
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        vis_frame = frame.copy()

        frame_preds = predictions.get(frame_id, [])
        detections = [Detection(frame_id, bbox) for bbox in frame_preds]
        tracker.update_tracks(detections, frame_id)

        for track in tracker.active_tracks:
            if track.get_last_frame_id() == frame_id:
                detection = track.get_last_detection()
                if detection:
                    x_min, y_min, x_max, y_max = detection.get_bb()
                    width = x_max - x_min
                    height = y_max - y_min
                    score = detection.get_score()
                    computed_results.append((frame_id, track.id, x_min, y_min, width, height, score))
                    vis_frame = cv2.rectangle(vis_frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0,255,0), 2)
                    vis_frame = cv2.putText(vis_frame, f"C:{track.id}", (int(x_min), int(y_min)-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    
        tree = ET.parse(gt_xml_path)
        root = tree.getroot()
        for track in root.findall('track'):
            track_id = track.attrib.get('id')
            for box in track.findall('box'):
                f = int(box.attrib.get('frame'))
                if f == frame_id:
                    xtl = float(box.attrib.get('xtl'))
                    ytl = float(box.attrib.get('ytl'))
                    xbr = float(box.attrib.get('xbr'))
                    ybr = float(box.attrib.get('ybr'))
                    vis_frame = cv2.rectangle(vis_frame, (int(xtl), int(ytl)), (int(xbr), int(ybr)), (0,0,255), 2)
                    vis_frame = cv2.putText(vis_frame, f"GT:{track_id}", (int(xtl), int(ytl)-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        vis_frames.append(vis_frame)
        frame_id += 1

    cap.release()

    tracker_results_path = os.path.join(tracker_folder, "tracker.txt")
    write_tracker_results(computed_results, tracker_results_path)

    trackeval_output_folder = os.path.join(output_folder, "trackeval_results")
    os.makedirs(trackeval_output_folder, exist_ok=True)
    
    cmd = [
        "python", "/Users/papallusqueti/Downloads/TrackEval-master/scripts/run_mot_challenge.py",
        "--TRACKERS_FOLDER", os.path.join(output_folder, "tracker"),
        "--GT_FOLDER", os.path.join(output_folder, "gt/seq01/gt"),
        "--OUTPUT_FOLDER", trackeval_output_folder,
        "--METRICS", "HOTA,IDF1",
        "--SEQMAP_FILE /Users/papallusqueti/mcv-c6-2025-team3/W2/output_task_6/gt/seqmaps/MOT17-train.txt"
    ]

    print("Running TrackEval...")
    subprocess.run(cmd, check=True)
    
    results_file = os.path.join(trackeval_output_folder, "TRACK_EVAL_results.txt")
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            trackeval_results = f.read()
        print("TrackEval Results:")
        print(trackeval_results)
    else:
        print("TrackEval results file not found.")

    print("Creating visualization GIF...")
    vis_frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in vis_frames]
    gif_path = os.path.join(output_folder, "evaluation.gif")
    imageio.mimsave(gif_path, vis_frames_rgb, fps=5)
    print(f"Visualization GIF saved to {gif_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tracking Evaluation with TrackEval")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file")
    parser.add_argument("--gt", type=str, required=True, help="Path to the ground truth XML file")
    parser.add_argument("--output", type=str, required=True, help="Output folder for results and visualization")
    args = parser.parse_args()
    run_evaluation(args.video, args.gt, args.output)