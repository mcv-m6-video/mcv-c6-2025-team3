import os
import numpy as np
import cv2
from ultralytics import YOLO
import torch
import json
from pathlib import Path
from sort import Sort, convert_x_to_bbox
import xml.etree.ElementTree as ET
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import imageio
import subprocess
import ptlflow
from ptlflow.utils.io_adapter import IOAdapter
from PIL import Image, ImageSequence
import torchvision.transforms as transforms
from fastreid.config import get_cfg
from fastreid.modeling import build_model
from fastreid.utils.checkpoint import Checkpointer
import sys


def detect_yoloft(model, frame, scale_factor, conf_th=0.5, roi_mask= None):
    if roi_mask is not None:
        img_height, img_width = roi_mask.shape

    results = model(frame, verbose=False)

    boxes = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])  #id
            conf = float(box.conf[0])  #score
            x1, y1, x2, y2 = map(float, box.xyxy[0])  #bbox
            x1 *= scale_factor
            y1 *= scale_factor
            x2 *= scale_factor
            y2 *= scale_factor

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            if (cls_id == 0 and conf >= conf_th):
                if roi_mask is not None :

                    #check roi mask
                    x1 = max(0, min(img_width - 1, x1))
                    y1 = max(0, min(img_height - 1, y1))
                    x2 = max(x1 + 1, min(img_width -1, x2))
                    y2 = max(y1 + 1, min(img_height -1, y2))
                    if roi_mask[y1, x1] > 200 and roi_mask[y1, x2] > 200 and roi_mask[y2, x1] > 200 and roi_mask[y2, x2] > 200:
                        boxes.append((x1, y1, x2, y2, cls_id, conf))

                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"car: {conf:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else: 
                    boxes.append((x1, y1, x2, y2, cls_id, conf))

                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"car: {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return boxes, frame
    


def save_json(file: dict, name: str):
    with open(name, "w") as f:
        json.dump(file, f)

def frames2gif(frames_list, output_gif, fps=30):
   with imageio.get_writer(output_gif, mode='I', duration=1000 / fps) as writer:
        for frame in frames_list:
            frame_resized = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            writer.append_data(frame_rgb)

def setup_model(config_path, model_weights):
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = build_model(cfg)
    model.eval()
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)
    return model, cfg

transform = transforms.Compose([
    transforms.Resize((256, 128)),  #ReId size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #imaenet norm
])

def extract_embedding(image, bbox, model, cfg):
    x, y, w, h = bbox
    cropped_image = image[int(y):int(y+h), int(x):int(x+w)]
    cropped_image = transform(Image.fromarray(cropped_image)).unsqueeze(0).to(cfg.MODEL.DEVICE)

    with torch.no_grad():
        embedding = model(cropped_image)

    return embedding.squeeze().cpu().numpy()


class KalmanFilter:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3, scale_factor = 1):
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.iou_threshold = float(iou_threshold)
        self.kalman_tracker = Sort(max_age = self.max_age, min_hits=self.min_hits, iou_threshold = self.iou_threshold)
        self.tracker_dict = {}
        self.n_frames = 1
        self.scale_factor = scale_factor
        self.image_width = 0
        self.image_height = 0
        config_path = "/home/danielpardo/fast-reid/configs/VeRi/sbs_R50-ibn.yml" 
        model_weights = "/home/danielpardo/fast-reid/models/veri_sbs_R50-ibn.pth"
        self.model, self.cfg = setup_model(config_path, model_weights)

    def next_frame(self):
        self.n_frames += 1

    def update(self, bbox_detection, frame):
        #update with Sort
        bboxes = bbox_detection[:, :4]
        conf_scores = bbox_detection[:, 4]
        class_ids = bbox_detection[:, 5]

        predicted_tracks = self.kalman_tracker.update(bboxes)

        if len(predicted_tracks) == 0:
            print(f"Failed to track objects in frame {self.n_frames}")

        self.update_tracker_dict(predicted_tracks, conf_scores, class_ids, frame)
    

    def update_tracker_dict(self, predicted_tracks, conf_scores, class_ids, frame):
        #print('Updating tracker dict...')
        if predicted_tracks.shape[0] == 0:
            return 
    
        kalman_predicted_bbox = predicted_tracks[:, 0:4]
        track_predicted_ids = predicted_tracks[:, 4]

        x_min, y_min, x_max, y_max, confs, classes, embeddings = {}, {}, {}, {}, {}, {}, {}
        for bbox, track_id, conf, class_id in zip(kalman_predicted_bbox, track_predicted_ids, conf_scores, class_ids):
            
            x_min_val = max(0, min(self.image_width - 1, bbox[0]))
            y_min_val = max(0, min(self.image_height - 1, bbox[1]))
            x_max_val = max(x_min_val + 1, min(self.image_width, bbox[2]))
            y_max_val = max(y_min_val + 1, min(self.image_height, bbox[3]))

            x_min[str(int(track_id))] = x_min_val
            y_min[str(int(track_id))] = y_min_val
            x_max[str(int(track_id))] = x_max_val
            y_max[str(int(track_id))] = y_max_val
            confs[str(int(track_id))] = conf
            classes[str(int(track_id))] = class_id

            embeddings[str(int(track_id))] = extract_embedding(frame, [x_min_val, y_min_val, x_max_val, y_max_val], self.model, self.cfg)

        self.tracker_dict[self.n_frames] = {
            'x_min': {k: float(v) for k, v in x_min.items()},
            'y_min': {k: float(v) for k, v in y_min.items()},
            'x_max': {k: float(v) for k, v in x_max.items()},
            'y_max': {k: float(v) for k, v in y_max.items()},
            'conf': {k: float(v) for k, v in confs.items()},
            'class_id': {k: int(v) for k, v in classes.items()},
            'embedding': {k: embeddings[k].tolist() for k in embeddings}
        }

        self.next_frame()
    
    def draw_tracking(self, frame):
        frame = frame.copy()

        for track in self.kalman_tracker.trackers:
            if track.time_since_update > 1:
                continue
            
            # Convert KalmanFilter x to bbox
            bbox = convert_x_to_bbox(track.kf.x).squeeze()

            # color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            # Draw bbox
            frame = cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), track.color, 4)

            # Draw text box
            frame = cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + 50), int(bbox[1] + 30)), track.color, -1)
            frame = cv2.putText(frame, str(track.id), (int(bbox[0]+5), int(bbox[1] + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), thickness=2)

            for detection in track.history:
                x_center = int((detection[0][0] + detection[0][2]) / 2)
                y_center = int((detection[0][1] + detection[0][3]) / 2)
                frame = cv2.circle(frame, (x_center, y_center), 5, track.color, -1)

        frame = cv2.rectangle(frame, (10, 10), (80, 40), (0, 0, 0), -1)
        frame = cv2.putText(frame, f"{self.n_frames}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(255, 255, 255), thickness=2) 

        return frame
    
    def execute(self, video_path, output_path, track_eval_path, roi_path, camera):

        predictions_file = f'{output_path}/predictions_yolov8n_{camera}.json'

        # Check if the predictions file exists
        if os.path.exists(predictions_file):
            print(f"Reading predictions from 'predictions_yolov8n_{camera}.json'...")
            with open(predictions_file, 'r') as f:
                predictions = json.load(f)
        else:
            print("Detecting objects using yolov8n...")
            model_path = "/home/danielpardo/c6/W3/yolov8n_ft.pt"
            model = YOLO(model_path)
            conf_th = 0.5

            frame_number = 0 
            predictions = {}
            frames_detection = []

            roi_mask = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)

            cap = cv2.VideoCapture(video_path)

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w > 2000:
                self.scale_factor = 2
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_number += 1

                new_w, new_h = int(w / self.scale_factor), int(h / self.scale_factor)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                predictions_frame, frame = detect_yoloft(model, frame, self.scale_factor, conf_th, roi_mask)

                predictions[frame_number] = predictions_frame
                frames_detection.append(frame)

            cap.release()

            # Save predictions to a JSON file
            with open(predictions_file, 'w') as f:
                json.dump(predictions, f, indent=4)

            predictions = json.loads(json.dumps(predictions))
    
        n_frames = len(predictions)


        cap = cv2.VideoCapture(video_path)

        self.image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_id = 1
        frames_tracking = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_id > n_frames:
                break 

            detections_current_frame = predictions.get(str(frame_id), [])
            # print('Detections', detections_current_frame)
            
            # Convert to array of [x_min, y_min, x_max, y_max, score]
            frame_boxes = []
            for x_min, y_min, x_max, y_max, class_id, score in detections_current_frame:
                bbox = [x_min, y_min, x_max, y_max, float(score), float(class_id)]
                frame_boxes.append(bbox)
            frame_boxes = np.array(frame_boxes)
            # print('Boxes', frame_boxes)

            if frame_boxes.shape[0] == 0:
                frame_boxes = np.empty((0, 6), dtype=np.float32) 

            # Update the KalmanFilter with the detections
            self.update(frame_boxes, frame)

            draw = self.draw_tracking(frame)
            frame_resized = cv2.resize(draw, (480, 270), interpolation=cv2.INTER_AREA)
            frames_tracking.append(frame_resized)

            self.prev_frame = frame.copy()
            frame_id += 1

        cap.release()

        frames2gif(frames_tracking, Path(f'{output_path}/tracking_{self.max_age}_{self.min_hits}_{self.iou_threshold}_{camera}.gif'))

        save_json(self.tracker_dict, f'{output_path}/tracker_dict_{self.max_age}_{self.min_hits}_{self.iou_threshold}_{camera}.json')

        print("Tracking with Kalman filter finished!")




def trim_gif(input_gif, output_gif, start_time=0, end_time=10):
    gif = Image.open(input_gif)

    frame_rate = 30
    
    start_frame = int((start_time * 1000) / frame_rate) 
    end_frame = int((end_time * 1000) / frame_rate) 
    
    frames = []
    
    for i, frame in enumerate(ImageSequence.Iterator(gif)):
        if i < start_frame:
            continue  # Skip frames before the start time
        if i >= end_frame:
            break  # Stop once we've reached the end time
        frames.append(frame.copy())  # Add frames within the time range

    if frames:
        frames[0].save(output_gif, save_all=True, append_images=frames[1:], duration=frame_rate, loop=0)
        print(f"Trimmed GIF saved to {output_gif}")
    else:
        print(f"No frames found in the specified time range.")

def object_tracking_in_all_sequence(dataset_path, output_folder, track_eval_path):
    for camera_sequence in sorted(os.listdir(dataset_path)):
        camera_sequence_path = os.path.join(dataset_path, camera_sequence)
        if os.path.isdir(camera_sequence_path):
            print(f"Processing camera sequence: {camera_sequence}")
            video_path = os.path.join(camera_sequence_path, 'vdo.avi')
            roi_path = os.path.join(camera_sequence_path, 'roi.jpg')
            if not os.path.isfile(video_path):
                print(f"Video not found: {video_path}")
                continue
            # print(f"Processing video: {video_path}")
            ground_truth_path = os.path.join(camera_sequence_path, 'gt', 'gt.txt')
            if not os.path.isfile(ground_truth_path):
                print(f"Ground truth not found: {ground_truth_path}")
                continue
            # print(f"Processing ground truth: {ground_truth_path}")

            max_age =9
            min_hits = 2
            iou_threshold = 0.1
            scale_factor = 1

            kalman_filter = KalmanFilter(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold, scale_factor=scale_factor)
            kalman_filter.execute(video_path, output_folder, track_eval_path, roi_path=roi_path, camera=camera_sequence)



def evaluate_tracking(track_eval_path):
    command = [
        "python", f"{track_eval_path}/scripts/run_mot_challenge.py",
        "--BENCHMARK", "custom",
        "--SPLIT_TO_EVAL", "train",
        "--TRACKERS_TO_EVAL", "kalman",
        "--METRICS", "HOTA", "Identity",
        "--USE_PARALLEL", "False",
        "--NUM_PARALLEL_CORES", "1",
        "--DO_PREPROC", "False" 
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    idf1_value = 0
    hota_value = 0

    lines = result.stdout.split("\n")
    for i, line in enumerate(lines):
        if "HOTA: kalman-pedestrian" in line:
            hota_value = lines[i+1].split()[1]
        if "Identity: kalman-pedestrian" in line:
            idf1_value = lines[i+1].split()[1]

    return float(hota_value), float(idf1_value)

def skip_frames_S01(cam_name, fps=10):
    cam_timestamps = {
        "c001": -0.300,
        "c002": 1.640,
        "c003": 2.049,
        "c004": 2.177,
        "c005": 2.235
    }
    
    skip_frames = round((cam_timestamps["c005"] * fps) - (cam_timestamps[cam_name] * fps))
    return skip_frames

def skip_frames_S03(cam_name, fps=10):
    cam_timestamps = {
        "c010": 8.715,
        "c011": 8.457,
        "c012": 5.879,
        "c013": 0,
        "c014": 5.042,
        "c015": 8.492
    }
    
    skip_frames = round((cam_timestamps["c010"] * fps) - (cam_timestamps[cam_name] * fps))
    return skip_frames

def skip_frames_S04(cam_name, fps=10):
    cam_timestamps = {
        "c016": 0,
        "c017": 14.318,
        "c018": 29.955,
        "c019": 26.979,
        "c020": 25.905,
        "c021": 39.973,
        "c022": 49.422,
        "c023": 45.716,
        "c024": 50.853,
        "c025": 50.263,
        "c026": 70.450,
        "c027": 85.097,
        "c028": 100.110,
        "c029": 125.788,
        "c030": 124.319,
        "c031": 125.033,
        "c032": 125.199,
        "c033": 150.893,
        "c034": 140.218,
        "c035": 165.568,
        "c036": 170.797,
        "c037": 170.567,
        "c038": 175.426,
        "c039": 175.644,
        "c040": 175.838
    }
    
    skip_frames = round((cam_timestamps["c040"] * fps) - (cam_timestamps[cam_name] * fps))
    return 0

def bbox2global(bbox, H):
    H_inv = np.linalg.inv(H)

    x1, y1, x2, y2 = bbox 

    x_c, y_c = (x1 + x2) / 2, (y1 + y2) / 2  #calculate center of bbox

    #trasnform to global system
    point = np.array([x_c, y_c, 1]).reshape(3, 1)
    transformed_point = H_inv @ point
    #normalization
    X, Y = transformed_point[0][0] / transformed_point[2][0], transformed_point[1][0] / transformed_point[2][0]

    return X, Y

import matplotlib.pyplot as plt

def convert2gps(dataset_path, output_path):
    max_age =9
    min_hits = 2
    iou_threshold = 0.1
    scale_factor = 1

    colors = ['b', 'g', 'r', 'c', 'm']

    aXs, aYs = [], []
    cams = []

    for i, camera_sequence in enumerate(sorted(os.listdir(dataset_path))):
        camera_sequence_path = os.path.join(dataset_path, camera_sequence)
        cams.append(camera_sequence)

        Xs, Ys = [], []

        if os.path.isdir(camera_sequence_path):

            H_path = os.path.join(camera_sequence_path, 'calibration.txt')

            with open(H_path, 'r') as f:
                lines = f.readlines()

            H_rows = lines[0].strip().split(';')
            H = np.array([list(map(float, row.split())) for row in H_rows])

            with open(f'{output_path}/tracker_dict_{max_age}_{min_hits}_{iou_threshold}_{camera_sequence}.json', 'r') as f:
                tracker_dict = json.load(f)

            output_file = f'{output_path}/gps_{max_age}_{min_hits}_{iou_threshold}_{camera_sequence}.txt'
            
            with open(output_file, "w") as f:
                for frame, detections in tracker_dict.items():
                    if int(frame) <= skip_frames_S04(camera_sequence):
                        continue

                    x_min = detections["x_min"]
                    y_min = detections["y_min"]
                    x_max = detections["x_max"]
                    y_max = detections["y_max"]
                    confs = detections.get("conf", {})
                    class_ids = detections.get("class_id", {})
                    embeddings = detections['embedding']


                    for track_id, x1 in x_min.items():
                        y1 = y_min[track_id]
                        x2 = x_max[track_id]
                        y2 = y_max[track_id]

                        embedding = embeddings[track_id]
                        
                        width = x2 - x1
                        height = y2 - y1

                        conf = confs.get(track_id, 1.0)
                        class_id = -1
                        visibility = 1

                        X, Y = bbox2global([x1, y1, x2, y2], H)

                        Xs.append(float(X))
                        Ys.append(float(Y))

                        f.write(f"{frame}, {track_id}, {x1:.2f}, {y1:.2f}, {width:.2f}, {height:.2f}, {conf}, {class_id}, {visibility}, {X:.10f}, {Y:.10f}, {embedding}\n")
            f.close()
        
        plt.scatter(Xs, Ys, label=f'Camera {camera_sequence}', color=colors[i % len(colors)], s=1)
        plt.xlabel("Longitude / GPS X")
        plt.ylabel("Latitude / GPS Y")
        plt.title("Multi-Camera Object Tracking in GPS Coordinates")
        plt.legend()
        plt.savefig(f"{output_path}/tracking_plot_{camera_sequence}.png", dpi=300, bbox_inches="tight")
        plt.close()
        aXs.append(Xs)
        aYs.append(Ys)

    plt.figure(figsize=(8, 6))

    for i, (Xs, Ys) in enumerate(zip(aXs, aYs)):
        plt.scatter(Xs, Ys, color=colors[i % len(colors)], s=1, label=f'{cams[i]}')

    plt.xlabel("Longitude / GPS X")
    plt.ylabel("Latitude / GPS Y")
    plt.title("Multi-Camera Object Tracking in GPS Coordinates")
    plt.legend() 
    plt.savefig(f"{output_path}/tracking_plot_{camera_sequence}_all.png", dpi=300, bbox_inches="tight")
    plt.close()



import shutil
def evaluate(dataset_path, output_path, track_eval_path):
    max_age =9
    min_hits = 2
    iou_threshold = 0.1
    scale_factor = 1

    for camera_sequence in sorted(os.listdir(dataset_path)):
        camera_sequence_path = os.path.join(dataset_path, camera_sequence)
        if os.path.isdir(camera_sequence_path):
            ground_truth_path = os.path.join(camera_sequence_path, 'gt', 'gt.txt')
            if not os.path.isfile(ground_truth_path):
                print(f"Ground truth not found: {ground_truth_path}")
                continue

            with open(f'{output_path}/tracker_dict_{max_age}_{min_hits}_{iou_threshold}_{camera_sequence}.json', 'r') as f:
                tracker_dict = json.load(f)

            output_file = f'{track_eval_path}/data/trackers/mot_challenge/custom-train/kalman/data/s03.txt'
            
            with open(output_file, "w") as f:
                for frame, detections in tracker_dict.items():
                    x_min = detections["x_min"]
                    y_min = detections["y_min"]
                    x_max = detections["x_max"]
                    y_max = detections["y_max"]
                    confs = detections.get("conf", {})
                    class_ids = detections.get("class_id", {})

                    for track_id, x1 in x_min.items():
                        y1 = y_min[track_id]
                        x2 = x_max[track_id]
                        y2 = y_max[track_id]
                        
                        width = x2 - x1
                        height = y2 - y1

                        conf = confs.get(track_id, 1.0)
                        class_id = -1
                        visibility = 1
                        
                        f.write(f"{frame}, {track_id}, {x1:.2f}, {y1:.2f}, {width:.2f}, {height:.2f}, {conf}, {class_id}, {visibility}\n")
            f.close()
            dst = f'{track_eval_path}/data/gt/mot_challenge/custom-train/s03/gt/gt.txt'
            shutil.copy(ground_truth_path, dst)

            hota, idf1 = evaluate_tracking(track_eval_path)
            
            print(output_path)
            print(f"{camera_sequence} HOTA: {hota:.4f} | IDF1: {idf1:.4f} | Max age: {max_age} | Min hits: {min_hits} | IoU: {iou_threshold:.2f}\n")
            print("-" * 90 + "\n")

import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from haversine import haversine

def save_mots_file(df, output_path):
    mots_df = df[["FRAME_ID", "TRACK_ID", "X", "Y", "W", "H", "CONFIDENCE", "CLASS_ID", "VISIBILITY"]]
    with open(output_path, "w") as f:
        for _, row in mots_df.iterrows():
            line = f"{int(row['FRAME_ID'])}, {int(row['TRACK_ID'])}, {int(row['X'])}, {int(row['Y'])}, {int(row['W'])}, {int(row['H'])}, {float(row['CONFIDENCE'])}, {int(row['CLASS_ID'])}, {float(row['VISIBILITY'])}\n"
            f.write(line)


def load_tracking_file(file_path):

    df = pd.read_csv(file_path, header=None, skip_blank_lines=True, sep=",", engine="python")

    data_columns = 11
    final_df = df.iloc[:, :data_columns]  
    embeddings = df.iloc[:, data_columns:].astype(str)
    embeddings = embeddings.apply(lambda row: " ".join(row).replace("[", "").replace("]", "").strip(), axis=1)
    column_names = ["FRAME_ID", "TRACK_ID", "X", "Y", "W", "H", "CONFIDENCE", "CLASS_ID", "VISIBILITY", "LAT", "LON"]
    final_df.columns = column_names 

    final_df["EMBEDDING"] = embeddings.apply(lambda x: np.array([float(i) for i in x.split()], dtype=np.float32))

    return final_df


def haversine_distance(lat1, lon1, lat2, lon2):
    return haversine((lat1, lon1), (lat2, lon2)) *1000

def refine_tracking_ids(cams, gps_w=0.5, emb_w=0.5, frame_window=3, gps_th = 1, score_th = 0.7):

    for cam, df in cams.items():
        for idx, row in df.iterrows():
            frame_id, track_id, gps_lat, gps_lon, emb = row["FRAME_ID"], row["TRACK_ID"], row["LAT"], row["LON"], row["EMBEDDING"]

            track_id_frame =  df[(df["FRAME_ID"] == frame_id)].copy()["TRACK_ID"].unique()

            #select frames in the window, only past frames
            df_window = df[(df["FRAME_ID"] >= frame_id - frame_window) & (df["FRAME_ID"] < frame_id)].copy()

            if df_window.empty:
                continue  #firsts frames

            #distances between current frame and past frames (in the window)
            df_window["GPS_DIST"] = df_window.apply(lambda x: haversine_distance(x["LAT"], x["LON"], gps_lat, gps_lon), axis=1)

            #distance from currente frame
            df_window["FRAME_DIFF"] = frame_id - df_window["FRAME_ID"]
            df_window["FRAME_WEIGHT"] = np.exp(-0.5 * df_window["FRAME_DIFF"])  #penalty old frames, exponential decay function, 0 is the best (1)

            #check distance
            valid_matches = df_window[df_window["GPS_DIST"] < gps_th] #max distance in m

            if not valid_matches.empty:
                #embedding similarities
                emb_list = np.vstack(valid_matches["EMBEDDING"].values)
                sim_scores = cosine_similarity(emb.reshape(1, -1), emb_list)[0]

                #penalty big distances + old frames weight
                match_scores = (emb_w * sim_scores) + (gps_w * ((0.5 * np.exp(-valid_matches["GPS_DIST"].values)) + (0.5*valid_matches["FRAME_WEIGHT"].values)))

                best_match_idx = np.argmax(match_scores)
                best_match_track = valid_matches.iloc[best_match_idx]["TRACK_ID"]

                if match_scores[best_match_idx] > score_th and best_match_track != track_id and best_match_track not in track_id_frame:  
                    df.at[idx, "TRACK_ID"] = best_match_track

    return cams

def camera_multitracking(dataset_path, output_path, track_eval_path):
    max_age =9
    min_hits = 2
    iou_threshold = 0.1
    scale_factor = 1

    cams = {}
    
    for camera_sequence in sorted(os.listdir(dataset_path)):
        file_path = f'{output_path}/gps_{max_age}_{min_hits}_{iou_threshold}_{camera_sequence}.txt'
        cams[f"{camera_sequence}"] = load_tracking_file(file_path)

    #normalize embeddings all together with l2 norm
    df_all = pd.concat(cams.values(), ignore_index=True)

    embeddings = np.stack(df_all["EMBEDDING"].values)
    embeddings_normalized = normalize(embeddings, norm="l2")

    df_all["EMBEDDING"] = list(embeddings_normalized)

    df_splitted = np.split(df_all, np.cumsum([len(df) for df in cams.values()])[:-1])  

    cams_normalized = {cam: df_splitted[i] for i, cam in enumerate(cams.keys())}

    #print(cams_normalized['c001'][["FRAME_ID", "TRACK_ID", "X", "Y", "W", "H", "CLASS_ID", "VISIBILITY"]])
    #print(np.linalg.norm(cams_normalized['c001'].iloc[1]['EMBEDDING']))  #if ok shoud be ≈ 1.0
    #print(np.linalg.norm(cams_normalized['c005'].iloc[1]['EMBEDDING']))


    #########################################################################################################


    th_distance = 0.75

    #cams_refined = refine_tracking_ids(cams_normalized, gps_w=0.5, emb_w=0.5, frame_window=3, gps_th = 1, score_th = 0.7)
    cams_refined = cams_normalized

    first_frames = {cam: df["FRAME_ID"].min() for cam, df in cams_refined.items()}

    global_id_map = {} #should be {(cam, track_id): global_id}
    start_global_id = 9999
    next_global_id = start_global_id

    final_df = {cam: [] for cam in cams_refined}

    while any(first_frames[cam] <= cams_refined[cam]["FRAME_ID"].max() for cam in cams_refined):
        frame_data = {}
        for cam, df in cams_refined.items():
            if first_frames[cam] > df["FRAME_ID"].max():
                continue  #end of frames od the cam

            current_frame = df[df["FRAME_ID"] == first_frames[cam]]

            if not current_frame.empty:
                frame_data[cam] = current_frame #current frmae

        if len(frame_data) > 1: #at least 2 cam with detectios

            matched_tracks = {cam: set() for cam in frame_data.keys()}

            #matching based on distance between embeddings
            for cam1, df1 in frame_data.items():

                potential_matches = []

                for _, row1 in df1.iterrows():
                    track_id1 = row1["TRACK_ID"]
                    emb1 = row1["EMBEDDING"]

                    if track_id1 in matched_tracks[cam1]:#unified yet in this frame
                        continue

                    if (cam1, track_id1) in global_id_map: #unified yet
                        continue  

                    
                    for cam2, df2 in frame_data.items():
                        if cam1 == cam2:
                            continue

                        for _, row2 in df2.iterrows():
                            track_id2 = row2["TRACK_ID"]
                            emb2 = row2["EMBEDDING"]

                            if track_id2 in matched_tracks[cam2]:
                                continue

                            if (cam2, track_id2) in global_id_map: #unified yet
                                continue  


                            sim_distance = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]

                            if sim_distance > th_distance:
                                potential_matches.append((track_id1, cam2, track_id2, sim_distance))
                
                

                if len(potential_matches) >= 1:
                    potential_matches.sort(key=lambda x: x[3], reverse=True)
                    final_matches = []
                    seen = [] 

                    for (track1, cam2, track2, dist) in potential_matches:
                        current_tuple = (track1, cam2, track2)

                        is_duplicate = any(
                            sum((t1 == track1, c2 == cam2, t2 == track2)) >= 2  # Cuenta coincidencias
                            for t1, c2, t2 in seen
                        )

                        if not is_duplicate:
                            final_matches.append((track1, cam2, track2, dist))
                            seen.append((track1, cam2, track2)) 

                    cam_matched = []
                    global1 = []
                    global2 = []
                    new_match = []

                    for (track1, cam2, track2, dist) in final_matches:
                        if (cam1, track1) in global_id_map:
                            global1.append(cam1, track1)
                        elif (cam2, track2) in global_id_map:
                            global2.append(cam2, track2)
                        else:
                            if track1 in matched_tracks[cam1]:
                                continue
                            if track2 in matched_tracks[cam2]:
                                continue
                            new_match.append((cam1, track1, cam2, track2))
                            matched_tracks[cam1].add(track1)
                            matched_tracks[cam2].add(track2)

                    if len(new_match) > 0:
                        for (cam1, track1, cam2, track2) in new_match:
                            if (cam1, track1) in global_id_map:
                                global_id_map[(cam2, track2)] = global_id_map[(cam1, track1)] 
                            else:
                                global_id_map[(cam1, track1)] = next_global_id 
                                global_id_map[(cam2, track2)] = next_global_id 
                                next_global_id += 1
                    if len(global1) > 1:
                        for (cam1, track1) in global1:
                            global_id_map[(cam1, track1)] = global_id_map[(cam1, track1)] 

                    if len(global2) > 1:
                        for (cam2, track2) in global2:
                            global_id_map[(cam2, track2)] = global_id_map[(cam2, track2)] 
                else:
                    #no match, ignore track id
                    continue
        
        for cam, df in frame_data.items():
            df = df.copy()

            df["GLOBAL_TRACK_ID"] = df["TRACK_ID"].map(lambda x: global_id_map.get((cam, x), -1))
            df = df[df["GLOBAL_TRACK_ID"] >= start_global_id]  #only with global id, as it guarantee mathing in multiple cameras
            
            if not df.empty:
                df_selected = df[["FRAME_ID", "GLOBAL_TRACK_ID", "X", "Y", "W", "H", "CONFIDENCE", "CLASS_ID", "VISIBILITY"]].rename(columns={"GLOBAL_TRACK_ID": "TRACK_ID"})
                final_df[cam].append(df_selected)


        for cam in cams_refined.keys():
            first_frames[cam] += 1
        
    final_df_concatenated = {cam: pd.concat(final_df[cam], ignore_index=True) for cam in final_df if final_df[cam]}

    for cam in final_df_concatenated.keys():
        save_mots_file(final_df_concatenated[cam], f'{track_eval_path}/data/trackers/mot_challenge/custom-train/kalman/data/s03.txt')
        save_mots_file(final_df_concatenated[cam], f'{output_path}/predict_{max_age}_{min_hits}_{iou_threshold}_{cam}.txt')
        camera_sequence_path = os.path.join(dataset_path, cam)
        ground_truth_path = os.path.join(camera_sequence_path, 'gt', 'gt.txt')
        dst = f'{track_eval_path}/data/gt/mot_challenge/custom-train/s03/gt/gt.txt'
        shutil.copy(ground_truth_path, dst)
        hota, idf1 = evaluate_tracking(track_eval_path)

        print(output_path)
        print(f'th_distance:{th_distance}')
        print(f"{cam} HOTA: {hota:.4f} | IDF1: {idf1:.4f} | Max age: {max_age} | Min hits: {min_hits} | IoU: {iou_threshold:.2f}\n")
        print("-" * 90 + "\n")
