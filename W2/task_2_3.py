import sys
sys.path.insert(0, '/Users/papallusqueti/Downloads/TrackEval-master')

import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from trackeval.metrics.hota import HOTA
from trackeval.metrics.identity import Identity

def parse_mot_file(mot_file_path):
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
def parse_gt_file(gt_path):
    """Parse the ground truth XML file and return a dictionary mapping frame numbers to detections."""
    gt_detections = {}
    tree = ET.parse(gt_path)
    root = tree.getroot()
    for track in root.findall('track'):
        track_id = int(track.attrib.get('id'))
        for box in track.findall('box'):
            frame = int(box.attrib.get('frame'))
            xtl = float(box.attrib.get('xtl'))
            ytl = float(box.attrib.get('ytl'))
            xbr = float(box.attrib.get('xbr'))
            ybr = float(box.attrib.get('ybr'))
            bbox = [xtl, ytl, xbr - xtl, ybr - ytl]
            gt_detections.setdefault(frame, []).append((track_id, bbox))
    return gt_detections

def read_tracker_file(tracker_path):
    """Read the tracker file (MOT format) and return a dictionary mapping frame numbers to detections."""
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
    """Build lists of IDs and similarity (IoU) matrices for each frame."""
    gt_ids_list = []
    tracker_ids_list = []
    similarity_scores_list = []
    
    # Collect all frames from both GT and tracker detections
    all_frames = sorted(set(gt_dets.keys()) | set(tracker_dets.keys()))
    
    # Build mapping of unique IDs to contiguous indices.
    all_gt_ids = sorted({tid for dets in gt_dets.values() for tid, _ in dets})
    all_tracker_ids = sorted({tid for dets in tracker_dets.values() for tid, _ in dets})
    gt_id_map = {tid: idx for idx, tid in enumerate(all_gt_ids)}
    tracker_id_map = {tid: idx for idx, tid in enumerate(all_tracker_ids)}
    
    for frame in all_frames:
        frame_gt = gt_dets.get(frame, [])
        frame_tracker = tracker_dets.get(frame, [])
        # Map the IDs to contiguous indices
        gt_ids = np.array([gt_id_map[tid] for tid, _ in frame_gt], dtype=int)
        tracker_ids = np.array([tracker_id_map[tid] for tid, _ in frame_tracker], dtype=int)
        gt_ids_list.append(gt_ids)
        tracker_ids_list.append(tracker_ids)
        
        # Compute similarity matrix based on IoU.
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

def run_metrics_from_files(gt_file, tracker_file, output_folder):
    # Read GT and tracker detections using the MOT parser
    gt_dets = parse_mot_file(gt_file)
    tracker_dets = read_tracker_file(tracker_file)  # assuming this already reads MOT files
    
    # Build the evaluation data dictionary as before
    data = group_ids_and_similarity(gt_dets, tracker_dets)
    
    # Compute metrics
    hota_metric = HOTA()
    id_metric = Identity()
    hota_results = hota_metric.eval_sequence(data)
    id_results = id_metric.eval_sequence(data)
    
    print("HOTA results:")
    print(hota_results)
    print("IDF1 results:")
    print(id_results)
    
    return {'hota': hota_results, 'idf1': id_results}

def compare_metrics_across_thresholds(base_output_folder, iou_thresholds=[0.1, 0.3, 0.5, 0.7]):
    """
    For each IoU threshold folder under base_output_folder (e.g., min_iou_0.1, etc.),
    compute the HOTA and IDF1 metrics from pre-computed GT and tracker files, then plot them.
    """
    import matplotlib.pyplot as plt

    hota_values = []
    idf1_values = []
    
    for iou in iou_thresholds:
        print(f"Processing metrics for min_iou = {iou}")
        # Construct paths to the GT and tracker files for each threshold folder.
        output_subfolder = base_output_folder / f"min_iou_{iou}"
        gt_file = output_subfolder / "gt" / "seq01" / "gt.txt"
        tracker_file = output_subfolder / "tracker" / "seq01" / "tracker.txt"
        
        # Run the metrics from files (using your existing function).
        results = run_metrics_from_files(gt_file, tracker_file, output_subfolder)
        
        # Extract the overall HOTA at alpha=0 and the IDF1 metric.
        hota_val = results["hota"].get("HOTA(0)", None)
        idf1_val = results["idf1"].get("IDF1", None)
        
        if hota_val is None or idf1_val is None:
            print(f"Warning: Missing metrics for min_iou = {iou}")
        else:
            hota_values.append(hota_val)
            idf1_values.append(idf1_val)

    # Plot both metrics versus the min_iou thresholds.
    plt.figure(figsize=(8, 6))
    plt.plot(iou_thresholds, hota_values, marker='o', label='HOTA(0)')
    plt.plot(iou_thresholds, idf1_values, marker='s', label='IDF1')
    plt.xlabel('min_iou Threshold')
    plt.ylabel('Metric Score')
    plt.title('Comparison of HOTA(0) and IDF1 for Different min_iou Thresholds')
    plt.legend()
    plt.grid(True)
    comparison_plot_path = base_output_folder / "metrics_comparison.png"
    plt.savefig(comparison_plot_path)
    plt.show()
    print(f"Comparison plot saved to {comparison_plot_path}")

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Task 2.3: Object tracking evaluation from files")
    parser.add_argument("--base_output", type=str, default="output_task_6", 
                        help="Base output folder where the min_iou_* folders are stored")
    args = parser.parse_args()

    base_folder = Path(args.base_output).absolute()
    compare_metrics_across_thresholds(base_folder)