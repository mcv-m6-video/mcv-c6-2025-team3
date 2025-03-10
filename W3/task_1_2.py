import os
import numpy as np
import cv2
from ultralytics import YOLO
import torch
import json
from pathlib import Path
from sortOF import Sort, convert_x_to_bbox
import xml.etree.ElementTree as ET
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import imageio
import subprocess
import ptlflow
from ptlflow.utils.io_adapter import IOAdapter

def annonations2mot(gt_path, output_path):
    tree = ET.parse(gt_path)
    root = tree.getroot()

    with open(output_path, "w") as f:
        for track in root.findall('track'):
            label = track.get("label")      
            if label != 'car':
                continue
            
            track_id = int(track.attrib.get('id'))
            for box in track.findall('box'):
                
                frame = int(box.attrib.get('frame'))+1
                xtl = float(box.attrib.get('xtl'))
                ytl = float(box.attrib.get('ytl'))
                xbr = float(box.attrib.get('xbr'))
                ybr = float(box.attrib.get('ybr'))
                width = xbr - xtl
                height = ybr - ytl
                
                conf = 1.0
                class_id = 0
                visibility = 1 
                f.write(f"{frame}, {track_id}, {xtl:.2f}, {ytl:.2f}, {width:.2f}, {height:.2f}, {conf}, {class_id}, {visibility}\n")

def create_folder_structure(trackEval_base_path):
    gt_folder = Path(f'{trackEval_base_path}/data/gt/mot_challenge/custom-train/s03/gt')
    gt_folder.mkdir(exist_ok=True, parents=True)

    with open(f'{trackEval_base_path}/data/gt/mot_challenge/custom-train/s03/seqinfo.ini', "w") as f:
        seqinfo_content = f"""[Sequence]
name=s03
seqLength=3000"""

        f.write(seqinfo_content)

    seqmaps_folder = Path(f'{trackEval_base_path}/data/gt/mot_challenge/seqmaps')
    seqmaps_folder.mkdir(exist_ok=True, parents=True)

    with open(f'{seqmaps_folder}/custom-train.txt', "w") as f:
        f.write(f"name\ns03")

    predict_folder = Path(f'{trackEval_base_path}/data/trackers/mot_challenge/custom-train/kalman/data')
    predict_folder.mkdir(exist_ok=True, parents=True)

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

def generate_optical_flow_video_gif(output_folder, video_path):

    model = optical_flow_model('maskflownet')

    frames_flow_color = []
    prev_frame = None

    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break 

        count+=1
        if prev_frame is not None:
            flow = compute_optical_flow(prev_frame, frame, model)
            flow_color = flow_to_color(flow[..., 0], flow[..., 1])
            frame_resized = cv2.resize(flow_color, (320, 180), interpolation=cv2.INTER_AREA)

            frames_flow_color.append(frame_resized)
    
        prev_frame = frame

    cap.release()
    frames2gif(frames_flow_color, Path(f'{output_folder}/optical_flow_video.gif'))


def detect_yoloft(model, frame, conf_th=0.5):

    results = model(frame, verbose=False)

    boxes = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])  #id
            conf = float(box.conf[0])  #score
            x1, y1, x2, y2 = map(float, box.xyxy[0])  #bbox

            if (cls_id == 0 and conf >= conf_th):
                boxes.append((x1, y1, x2, y2, cls_id, conf))

                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"car: {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return boxes, frame
    
def flow_to_color(flow_x, flow_y):
    h, w = flow_x.shape
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    magnitude, angle = cv2.cartToPolar(flow_x, flow_y)

    hsv[..., 0] = angle * 180 / np.pi / 2  # Hue: direction
    hsv[..., 1] = 255  # Saturation: max
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value: speed (scaled), normalized

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def optical_flow_model(option='maskflownet'):
    return ptlflow.get_model(f"{option}", ckpt_path='kitti')

def compute_optical_flow(image1, image2, model):
    #ensure rgb
    if len(image1.shape) == 2:  
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    io_adapter = IOAdapter(model, img1.shape[:2])
    data = io_adapter.prepare_inputs([img1, img2])

    output = model(data)

    flow = output['flows']
    flow = np.squeeze(flow)  
    flow = flow.permute(1, 2, 0)
    flow = flow.detach().cpu().numpy()

    return flow



def save_json(file: dict, name: str):
    with open(name, "w") as f:
        json.dump(file, f)

def frames2gif(frames_list, output_gif, fps=30):
   with imageio.get_writer(output_gif, mode='I', duration=1000 / fps) as writer:
        for frame in frames_list:
            frame_resized = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            writer.append_data(frame_rgb)

def predict_bbox_with_flow(bbox, flow):
    #fucntion to estimate bbox of the current frame usint past bbox and current optical flow
    x1, y1, x2, y2 = map(int, bbox)
    h, w = flow.shape[:2]

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    roi_flow = flow[y1:y2, x1:x2]
    if roi_flow.size == 0:
        return bbox

    flow_x = roi_flow[..., 0]
    flow_y = roi_flow[..., 1]

    if flow_x.size == 0 or flow_y.size == 0:
        return bbox

    dx = np.median(flow_x)
    dy = np.median(flow_y)

    #check image limits
    x1_new, y1_new = np.clip([x1 + dx, y1 + dy], [0, 0], [w - 1, h - 1])
    x2_new, y2_new = np.clip([x2 + dx, y2 + dy], [x1_new + 1, y1_new + 1], [w, h])

    return [x1_new, y1_new, x2_new, y2_new]




def iou(bboxA, bboxB):
    xA = max(bboxA[0], bboxB[0])
    yA = max(bboxA[1], bboxB[1])
    xB = min(bboxA[2], bboxB[2])
    yB = min(bboxA[3], bboxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (bboxA[2] - bboxA[0]) * (bboxA[3] - bboxA[1])
    boxBArea = (bboxB[2] - bboxB[0]) * (bboxB[3] - bboxB[1])
    iou_val = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou_val

def find_best_match(flow_bbox, detections, threshold=0.5):
    best_iou = 0.0
    best_idx = None
    for idx, det in enumerate(detections):
        x1, y1, x2, y2, _, _ = det
        iou_val = iou(flow_bbox, [x1, y1, x2, y2])
        if iou_val > threshold and iou_val > best_iou:
            best_iou = iou_val
            best_idx = idx
    return best_idx, best_iou

class KalmanFilterWithOpticalFlow:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3, optical_flow=False):
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.iou_threshold = float(iou_threshold)
        self.optical_flow = optical_flow
        self.kalman_tracker = Sort(max_age = self.max_age, min_hits=self.min_hits, iou_threshold = self.iou_threshold)
        self.tracker_dict = {}
        self.n_frames = 1
        self.model = optical_flow_model('maskflownet')
        self.image_width = 0
        self.image_height = 0
    
    def next_frame(self):
        self.n_frames += 1

    def update(self, bbox_detection, flow = None):
        #update with Sort
        bboxes = bbox_detection[:, :4]
        conf_scores = bbox_detection[:, 4]
        class_ids = bbox_detection[:, 5]

        predicted_tracks = self.kalman_tracker.update(bboxes, flow)


        if len(predicted_tracks) == 0:
            print(f"Failed to track objects in frame {self.n_frames}")
        else: #check for unmached tracks
            if self.optical_flow and flow is not None:
                unmatched_tracks = []
                for track in self.kalman_tracker.trackers:
                    if track.time_since_update == 1: #1 for this current frame
                        unmatched_tracks.append(track)
                
                #search bboxes from YOLO unused in Sort
                used_indices = []
                #for this, first we searh tracks that have iuo > iou_threshold with yolo
                #detected object to assume that it is a match
                for pred_track in predicted_tracks:
                    match_idx = None
                    for i, bbox in enumerate(bbox_detection):
                        iou_val = iou(pred_track[:4], bbox[:4])
                        if iou_val > self.iou_threshold:
                            match_idx = i
                            break
                    #if we dont have match it is unsused bbox
                    if match_idx is not None:
                        used_indices.append(match_idx)
                #add all uunused bboxes
                unused_detections = []
                for i, bbox in enumerate(bbox_detection):
                    if i not in used_indices:
                        unused_detections.append(bbox)

                matches = [] 
                #now we try to assign unmatched track to unused bbox using optical flow
                for track_obj in unmatched_tracks:
                    #previous estimated bbox, since in current frame is missing
                    prev_bbox = convert_x_to_bbox(track_obj.kf.x).squeeze()
                    #estimate hypothetical bbox in current frame using flow
                    flow_bbox = predict_bbox_with_flow(prev_bbox, flow)

                    #now we find a match between this estimated bbox and the best unused with a th
                    match_id, best_iou = find_best_match(flow_bbox, unused_detections, threshold=self.iou_threshold)
                    if match_id is not None:
                        matches.append((track_obj, match_id, best_iou))

                matches.sort(key=lambda x: x[2], reverse=True)

                used_detections = set()
                
                #assing past id to this track, manual update

                for track_obj, bbox_idx, _ in matches:
                    if bbox_idx not in used_detections:
                        bbox_found = unused_detections[bbox_idx]
                        track_obj.update(np.array(bbox_found[:4]))
                        used_detections.add(bbox_idx)

        self.update_tracker_dict(predicted_tracks, conf_scores, class_ids)
    

    def update_tracker_dict(self, predicted_tracks, conf_scores, class_ids):
        #print('Updating tracker dict...')
        if predicted_tracks.shape[0] == 0:
            return 
    
        kalman_predicted_bbox = predicted_tracks[:, 0:4]
        track_predicted_ids = predicted_tracks[:, 4]

        x_min, y_min, x_max, y_max, confs, classes = {}, {}, {}, {}, {}, {}
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

        self.tracker_dict[self.n_frames] = {
            'x_min': {k: float(v) for k, v in x_min.items()},
            'y_min': {k: float(v) for k, v in y_min.items()},
            'x_max': {k: float(v) for k, v in x_max.items()},
            'y_max': {k: float(v) for k, v in y_max.items()},
            'conf': {k: float(v) for k, v in confs.items()},
            'class_id': {k: int(v) for k, v in classes.items()}
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
    
    def execute(self, video_path, output_path, track_eval_path, c=None):

        predictions_file = f'{output_path}/predictions_yolov8n.json'

        # Check if the predictions file exists
        if os.path.exists(predictions_file):
            print("Reading predictions from 'predictions_yolov8n.json'...")
            with open(predictions_file, 'r') as f:
                predictions = json.load(f)
            n_frames = len(predictions)
        else:
            print("Detecting objects using yolov8n...")
            model_path = "/home/danielpardo/c6/W3/yolov8n_ft.pt"
            model = YOLO(model_path)
            conf_th = 0.5

            frame_number = 0 
            predictions = {}
            frames_detection = []

            cap = cv2.VideoCapture(video_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_number += 1
            
                predictions_frame, frame = detect_yoloft(model, frame, conf_th)

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

        prev_frame = None
        flow = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_id > n_frames:
                break 

            if self.optical_flow and prev_frame is not None:
                flow = compute_optical_flow(prev_frame, frame, self.model)

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
            self.update(frame_boxes, flow)

            draw = self.draw_tracking(frame)
            frame_resized = cv2.resize(draw, (320, 180), interpolation=cv2.INTER_AREA)
            frames_tracking.append(frame_resized)

            prev_frame = frame.copy()
            frame_id += 1

        cap.release()

        if c is not None:
            frames2gif(frames_tracking, Path(f'{output_path}/tracking_{self.max_age}_{self.min_hits}_{self.iou_threshold}_{self.optical_flow}_{c}.gif'))
        else:
            frames2gif(frames_tracking, Path(f'{output_path}/tracking_{self.max_age}_{self.min_hits}_{self.iou_threshold}_{self.optical_flow}.gif'))

        save_json(self.tracker_dict, f'{output_path}/kalman_tracker_dict.json')

        with open(f'{output_path}/kalman_tracker_dict.json', 'r') as f:
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
                    class_id = class_ids.get(track_id, 1)
                    visibility = 1
                    
                    f.write(f"{frame}, {track_id}, {x1:.2f}, {y1:.2f}, {width:.2f}, {height:.2f}, {conf}, {class_id}, {visibility}\n")

        print("Tracking with Kalman filter finished!")