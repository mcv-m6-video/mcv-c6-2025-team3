
import sys
sys.path.insert(0, '/Users/papallusqueti/Downloads/TrackEval-master')
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from trackeval.metrics.hota import HOTA
from trackeval.metrics.identity import Identity

def parse_gt_file(mot_file_path):
    """
    Parse a MOT format file.
    Each line is expected to be: 
      frame, track_id, x, y, width, height, score, class, -1, -1
    Returns a dictionary mapping each frame to a list of tuples: (track_id, [x, y, width, height])
    """
    detections = {}
    with open(mot_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue
            frame = int(parts[0])
            track_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            width = float(parts[4])
            height = float(parts[5])
            bbox = [x, y, width, height]
            detections.setdefault(frame, []).append((track_id, bbox))
    return detections

def read_tracker_file(tracker_path):
    """
    Read the tracker file (MOT format) and return a dictionary mapping frame numbers to detections.
    Each detection is a tuple: (track_id, [x, y, width, height]).
    Expected line format:
      frame, track_id, x, y, width, height, score, -1, -1, -1
    """
    tracker_detections = {}
    with open(tracker_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue
            frame = int(parts[0])
            track_id = int(parts[1])
            bbox = list(map(float, parts[2:6]))
            tracker_detections.setdefault(frame, []).append((track_id, bbox))
    return tracker_detections

def compute_iou(bbox1, bbox2):
    """Compute Intersection over Union for two bounding boxes."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def group_ids_and_similarity(gt_dets, tracker_dets):
    """
    Build lists for each frame:
      - gt_ids: an array of ground truth detection indices.
      - tracker_ids: an array of tracker detection indices.
      - similarity_scores: a matrix of IoU scores between GT and tracker detections.
    Also returns total unique counts.
    """
    gt_ids_list = []
    tracker_ids_list = []
    similarity_scores_list = []
    
    all_frames = sorted(set(gt_dets.keys()) | set(tracker_dets.keys()))
    all_gt_ids = sorted({tid for dets in gt_dets.values() for tid, _ in dets})
    all_tracker_ids = sorted({tid for dets in tracker_dets.values() for tid, _ in dets})
    gt_id_map = {tid: idx for idx, tid in enumerate(all_gt_ids)}
    tracker_id_map = {tid: idx for idx, tid in enumerate(all_tracker_ids)}
    
    for frame in all_frames:
        frame_gt = gt_dets.get(frame, [])
        frame_tracker = tracker_dets.get(frame, [])
        gt_ids = np.array([gt_id_map[tid] for tid, _ in frame_gt], dtype=int)
        tracker_ids = np.array([tracker_id_map[tid] for tid, _ in frame_tracker], dtype=int)
        gt_ids_list.append(gt_ids)
        tracker_ids_list.append(tracker_ids)
        
        if frame_gt and frame_tracker:
            sim_matrix = np.zeros((len(frame_gt), len(frame_tracker)))
            for i, (_, bbox_gt) in enumerate(frame_gt):
                for j, (_, bbox_tr) in enumerate(frame_tracker):
                    sim_matrix[i, j] = compute_iou(bbox_gt, bbox_tr)
        else:
            sim_matrix = np.empty((len(frame_gt), len(frame_tracker)))
        similarity_scores_list.append(sim_matrix)
    
    data = {
        'num_gt_dets': sum(len(ids) for ids in gt_ids_list),
        'num_tracker_dets': sum(len(ids) for ids in tracker_ids_list),
        'gt_ids': gt_ids_list,
        'tracker_ids': tracker_ids_list,
        'similarity_scores': similarity_scores_list,
        'num_gt_ids': len(all_gt_ids),
        'num_tracker_ids': len(all_tracker_ids)
    }
    return data

def main(gt_file, tracker_file):
    # Read the ground truth and tracker detections.
    gt_dets = parse_gt_file(gt_file)
    tracker_dets = read_tracker_file(tracker_file)
    # Build the data dictionary expected by the metric classes.
    data = group_ids_and_similarity(gt_dets, tracker_dets)
    
    # Instantiate and evaluate the metrics.
    hota_metric = HOTA()
    id_metric = Identity()
    hota_results = hota_metric.eval_sequence(data)
    id_results = id_metric.eval_sequence(data)
    
    print("HOTA results:")
    print(hota_results)
    print("IDF1 results:")
    print(id_results)
 
main('/Users/papallusqueti/mcv-c6-2025-team3/W2/output_task_6/gt/seq01/gt.txt', '/Users/papallusqueti/mcv-c6-2025-team3/W2/kalman_tracker.txt')