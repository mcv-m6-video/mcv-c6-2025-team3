import json
import os
from pathlib import Path
import random
import cv2
import numpy as np

from utils import frames2gif
from sort import Sort, convert_x_to_bbox
from task_1_1 import detect_cars_yolov8n

def save_json(file: dict, name: str):
    with open(name, "w") as f:
        json.dump(file, f)

class KalmanFilter:
    def __init__(self, max_age=30, iou_threshold=0.3):
        self.max_age = int(max_age)
        self.iou_threshold = iou_threshold
        self.kalman_tracker = Sort(max_age = int(max_age), iou_threshold = float(iou_threshold))
        
        self.tracker_dict = {}
        self.n_frames = 0
    
    def next_frame(self):
        self.n_frames += 1

    def update(self, bbox_detection):
        predicted_tracks = self.kalman_tracker.update(bbox_detection)
        # print('Predicted tracks for frame:', self.n_frames, ':', predicted_tracks)
        if len(predicted_tracks) == 0:
            print(f"Failed to track objects in frame {self.n_frames}")
        self.update_tracker_dict(predicted_tracks)
    
    def update_tracker_dict(self, predicted_tracks):
        print('Updating tracker dict...')
        if predicted_tracks.shape[0] == 0:
            return 
    
        kalman_predicted_bbox = predicted_tracks[:, 0:4]
        track_predicted_ids = predicted_tracks[:, 4]

        x_min, y_min, x_max, y_max = {},{},{},{}
        for bbox, id in zip(kalman_predicted_bbox, track_predicted_ids):
            x_min[str(int(id))] = bbox[0]
            y_min[str(int(id))] = bbox[1]
            x_max[str(int(id))] = bbox[2]
            y_max[str(int(id))] = bbox[3]

        self.tracker_dict[self.n_frames] = {
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max
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
            frame = cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + 20), int(bbox[3] - 50)), track.color, -1)
            frame = cv2.putText(frame, str(track.id), (int(bbox[0]), int(bbox[1] + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            for detection in track.history:
                x_center = int((detection[0][0] + detection[0][2]) / 2)
                y_center = int((detection[0][1] + detection[0][3]) / 2)
                frame = cv2.circle(frame, (x_center, y_center), 5, track.color, -1)
                
        return frame
    
    def execute(self, video_path, output_path, generate_video=True, show_tracking=True):

        predictions_file = 'predictions_yolov8n.json'

        # Check if the predictions file exists
        if os.path.exists(predictions_file):
            print("Reading predictions from 'predictions_yolov8n.json'...")
            with open(predictions_file, 'r') as f:
                predictions = json.load(f)
            n_frames = len(predictions)
        else:
            print("Detecting objects using yolov8n...")
            predictions = detect_cars_yolov8n(video_path, output_path)
            
            # Save predictions to a JSON file
            with open(predictions_file, 'w') as f:
                json.dump(predictions, f)
            
            n_frames = len(predictions)

        print(predictions)

        cap = cv2.VideoCapture(video_path)

        if generate_video:
            file_name = str(output_path / 'output.avi')
            codec = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(file_name, codec, fps, (frame_width, frame_height))
        
        frame_id = 0
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_id >= n_frames:
                break 

            detections_current_frame = predictions.get(str(frame_id), [])
            # print('Detections', detections_current_frame)
            
            # Convert to array of [x_min, y_min, x_max, y_max, score]
            frame_boxes = []
            for x_min, y_min, x_max, y_max, id, score in detections_current_frame:
                bbox =  [x_min, y_min, x_max, y_max, score]
                frame_boxes.append(bbox)
            frame_boxes = np.array(frame_boxes)
            # print('Boxes', frame_boxes)

            # Update the KalmanFilter with the detections
            self.update(frame_boxes)

            draw = self.draw_tracking(frame)
            if show_tracking:
                cv2.imshow('Tracking', draw)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if generate_video:
                out.write(draw)

            frame_id += 1
            frames.append(draw)

        frames2gif(frames, Path('output_task_5') / 'kalman_tracking.gif')
        save_json(self.tracker_dict, output_path / 'kalman_tracker_dict.json')
        cap.release()
        if generate_video:
            out.release()
        cv2.destroyAllWindows()
        
        print("Tracking with Kalman filter finished!")

def object_tracking_by_kalman_filter(video_path, output_folder):
    kalman_filter = KalmanFilter()
    kalman_filter.execute(video_path, output_folder)