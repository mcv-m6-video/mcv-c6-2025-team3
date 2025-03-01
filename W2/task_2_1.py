import os
import cv2
import numpy as np
from task_1_1 import detect_cars_yolov8n
from typing import List, Tuple

class Track:
    def __init__(self, track_id: int, first_detection, first_frame_id: int):
        self.id = track_id
        self.color = self._generate_color()
        self.detections: List = [first_detection]
        self.frames: List[int] = [first_frame_id]

    @staticmethod
    def _generate_color() -> Tuple[int, int, int]:
        return tuple(np.random.randint(0, 256, 3))

    def add_detection(self, detection, frame_id: int):
        self.detections.append(detection)
        self.frames.append(frame_id)

    def get_last_detection(self):
        return self.detections[-1] if self.detections else None

    def get_last_frame_id(self) -> int:
        return self.frames[-1] if self.frames else -1

    def get_color(self):
        return (int(self.color[0]), int(self.color[1]), int(self.color[2]))
    
    def get_track_info(self):
        return {
            "id": self.id,
            "color": self.color,
            "detections": self.detections,
            "frames": self.frames,
        }

class Tracker:
    def __init__(self, min_iou: float, max_no_detect: int):
        self.min_iou = min_iou
        self.max_no_detect = max_no_detect

        self.n_total_tracks = 0
        self.active_tracks: List[Track] = []
        self.ended_tracks: List[Track] = []

    def create_track(self, frame_id: int, detection) -> Track:
        """Create a new track"""
        new_track = Track(self.n_total_tracks, detection, frame_id)
        self.n_total_tracks += 1
        self.active_tracks.append(new_track)
        return new_track

    def get_tracks(self):
        return self.active_tracks

    def update_tracks(self, new_detections: List, frame_id: int):
        """Update existing tracks or creates new ones if needed"""
        # Remove stale tracks
        self.active_tracks = [
            track for track in self.active_tracks
            if frame_id - track.get_last_frame_id() <= self.max_no_detect
        ]

        # Match new detections to existing tracks
        for detection in new_detections:
            best_track = None
            best_iou = -1

            for track in self.active_tracks:
                last_detection = track.get_last_detection()
                iou = last_detection.compute_iou(detection)

                if iou >= self.min_iou and iou > best_iou:
                    best_track, best_iou = track, iou

            if best_track:
                best_track.add_detection(detection, frame_id)
            else:
                self.create_track(frame_id, detection)

        print(f"Active Tracks: {len(self.active_tracks)}, Frame: {frame_id}")

class Detection():
    def __init__(self, frame_id, bbox):
        """
        Initializes a Detection object.

        Parameters:
        - frame_id (int): The frame number this detection belongs to.
        - bbox (tuple): A tuple (x_min, y_min, x_max, y_max, class_id, confidence).
        """
        self.frame_id = frame_id

        # Extract bounding box values
        self.box_x_min = bbox[0]
        self.box_y_min = bbox[1]
        self.box_x_max = bbox[2]
        self.box_y_max = bbox[3]
        self.class_id = bbox[4]  # Object class
        self.score = bbox[5]  # Confidence score

    def get_bb(self):
        """Returns the bounding box as (x_min, y_min, x_max, y_max)."""
        return self.box_x_min, self.box_y_min, self.box_x_max, self.box_y_max
    
    def get_score(self):
        """Returns the confidence score."""
        return self.score

    def get_class_id(self):
        """Returns the class ID."""
        return self.class_id

    def compute_iou(self, other):
        """Computes the Intersection over Union (IoU) with another detection."""
        x_left = max(self.box_x_min, other.box_x_min)
        y_top = max(self.box_y_min, other.box_y_min)
        x_right = min(self.box_x_max, other.box_x_max)
        y_bottom = min(self.box_y_max, other.box_y_max)

        intersection_area = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1)

        square1_area = (self.box_x_max - self.box_x_min + 1) * (self.box_y_max - self.box_y_min + 1)
        square2_area = (other.box_x_max - other.box_x_min + 1) * (other.box_y_max - other.box_y_min + 1)

        union_area = square1_area + square2_area - intersection_area

        iou = intersection_area / union_area if union_area > 0 else 0
        return iou


def object_tracking_by_overlap(video_path,output_folder):
    print("Detecting objects using yolov8n...")
    # Object detection using YOLOv8n
    predictions = detect_cars_yolov8n(video_path, output_folder) #
    # print(predictions)

    print("Initializing tracker...")
    track_updater = Tracker(min_iou=0.1, max_no_detect=10)

    n_frames = len(predictions)
    cap = cv2.VideoCapture(video_path)

    frame_id = 0
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break 

        if frame_id >= n_frames:  
            break  
        
        # Convert predictions to `Detection` objects
        detections = [Detection(frame_id, bbox) for bbox in predictions.get(frame_id, [])] #(x_min, y_min, x_max, y_max, class_id, confidence)[]
        track_updater.update_tracks(detections, frame_id)
        frame_tracks = track_updater.get_tracks()
        print(f"Frame {frame_id}: {len(frame_tracks)} active tracks\n")

        # Draw bounding boxes and track IDs
        for frame_track in frame_tracks:
            if frame_track.get_last_frame_id() == frame_id:
                detection = frame_track.get_last_detection()
                bb_color = frame_track.get_color()
                if detection:
                    x_min, y_min, x_max, y_max = detection.get_bb()

                # Draw bbox
                img = cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), bb_color, thickness=4)

                # Draw ID label
                id_label = f"ID: {frame_track.get_track_info()['id']}"
                label_size, _ = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_bg_end = (int(x_min) + label_size[0] + 20, int(y_min) - label_size[1] - 20)
                img = cv2.rectangle(img, (int(x_min), int(y_min) - 5), label_bg_end, bb_color, -1)

                # Draw track ID text
                img = cv2.putText(img, id_label, (int(x_min) + 10, int(y_min) - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Save the frame with bounding boxes
        output_path = os.path.join(output_folder, f"frame_{frame_id}.png")
        cv2.imwrite(output_path, img)

        frame_id += 1

    cap.release()  # Release the video file
    print("Tracking by overlap completed!")
