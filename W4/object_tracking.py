import cv2
import json
import numpy as np
import os
from pathlib import Path

from utils import frames2gif, save_json
from detect_cars_yolo import detect_cars_yolov8n
from sort import Sort, convert_x_to_bbox


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
    
    def execute(self, camera_sequence, video_path, output_path, generate_video=True, show_tracking=True):

        predictions_file = output_path / f'predictions_yolov8n_{camera_sequence}.json'

        # Check if the predictions file exists
        if os.path.exists(predictions_file):
            print(f"Reading predictions from 'predictions_yolov8n_{camera_sequence}.json'...")
            with open(predictions_file, 'r') as f:
                predictions = json.load(f)
        else:
            print(f"Detecting objects using yolov8n in camera sequence {camera_sequence}...")
            predictions = detect_cars_yolov8n(camera_sequence, video_path, output_path)

            # Save predictions to a JSON file
            with open(predictions_file, 'w') as f:
                json.dump(predictions, f)
            
        # print(predictions)
        n_frames = len(predictions)

        cap = cv2.VideoCapture(video_path)

        if generate_video:
            file_name = str(output_path / f'output_{camera_sequence}.avi')
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
            
            # Convert to array of [x_min, y_min, x_max, y_max, score]
            frame_boxes = []
            for x_min, y_min, x_max, y_max, id, score in detections_current_frame:
                bbox =  [x_min, y_min, x_max, y_max, score]
                frame_boxes.append(bbox)
            frame_boxes = np.array(frame_boxes)

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

        frames2gif(frames, Path(output_path) / f'kalman_tracking_{camera_sequence}.gif')
        save_json(self.tracker_dict, output_path / f'kalman_tracker_dict_{camera_sequence}.json')
        cap.release()
        if generate_video:
            out.release()
        cv2.destroyAllWindows()
        
        print("Tracking with Kalman filter finished!")

def object_tracking_in_all_sequence(dataset_path, output_folder):
    for camera_sequence in sorted(os.listdir(dataset_path)):
        camera_sequence_path = os.path.join(dataset_path, camera_sequence)
        if os.path.isdir(camera_sequence_path):
            print(f"Processing camera sequence: {camera_sequence}")
            video_path = os.path.join(camera_sequence_path, 'vdo.avi')
            if not os.path.isfile(video_path):
                print(f"Video not found: {video_path}")
                continue
            # print(f"Processing video: {video_path}")
            ground_truth_path = os.path.join(camera_sequence_path, 'gt', 'gt.txt')
            if not os.path.isfile(ground_truth_path):
                print(f"Ground truth not found: {ground_truth_path}")
                continue
            if camera_sequence in ['c003']:
                continue
            # print(f"Processing ground truth: {ground_truth_path}")
            kalman_filter = KalmanFilter()
            kalman_filter.execute(camera_sequence, video_path, output_folder)