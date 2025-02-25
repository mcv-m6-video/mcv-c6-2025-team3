import numpy as np
import cv2
import pickle
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import io
from contextlib import redirect_stdout
from utils import read_video, split_video_25_75

def compute_mean_and_std(frames):
    """Compute mean and variance of pixels in the first 25% of the 
    test sequence to model background

    Parameters
        frames_25 : np.ndarray([n_frames, 1080, 1920, 1])

    Returns
        mean : np.ndarray([1080, 1920])
        std : np.ndarray([1080, 1920])"""

    # Stack frames into a 3D array: (num_frames, height, width)
    frames_stack = np.stack(frames, axis=0)

    print("Computing mean and std...")
    mean = np.mean(frames_stack, axis=0)
    std = np.std(frames_stack, axis=0)

    return mean, std

def segment_foreground(frames, mean, std, alpha):
    """Return the estimation of the foreground using the reting 75% of the frames

     Returns
        foreground_background : np.ndarray([n_frames, 1080, 1920, 3], dtype=bool)
    """
    print("Segmenting foreground...")
    #implementar pseudocodigo slide 22: week1 instructions
    foreground = np.abs((frames-mean) >= alpha * (std + 2))
    return foreground.astype(bool)

def segment_foreground_chunks(frames, mean, std, alpha):
    n_chunks = 5
    chunks = np.array_split(frames, n_chunks)
    foreground_results = []

    for chunk in chunks:
        foreground = np.abs((frames-mean) >= alpha * (std + 2))
        foreground_results.append(foreground)

    return np.concatenate(foreground_results, axis=0).astype(bool)

# TASK 1.1: Gaussian modeling
def gaussian_modeling(video_path, annotations_path, alpha):

    # Read video to get frames from it
    color_frames, gray_frames = read_video(video_path)

    # Separate video in first 25% and second 75%
    color_frames_25, color_frames_75 = split_video_25_75(color_frames)
    gray_frames_25, gray_frames_75 = split_video_25_75(gray_frames)

    # Compute mean and variance of pixels in the first 25% frames
    mean, std = compute_mean_and_std(gray_frames_25)

    # Segment foreground
    foreground_segmented = segment_foreground_chunks(gray_frames_75, mean, std, alpha)

    return foreground_segmented, color_frames_75, len(color_frames_25)

def compute_bbox(foreground_segmented, frames, alpha, idx_frame):

    foreground_gray = (foreground_segmented * 255).astype(np.uint8)
    color_frame = frames.copy()

    output_path = f'bbox_task1_{alpha}.avi'
    fps = 30 
    frame_size = (color_frame.shape[2], color_frame.shape[1])

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

    bbox_dict = {}

    for idx, gray_frame in enumerate(foreground_gray):
        frame = cv2.medianBlur(gray_frame, 5)
        kernel = np.ones((5,5), np.uint8)
        frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
        frame = cv2.dilate(frame, kernel, iterations=6)
        frame = cv2.erode(frame, kernel, iterations=4)
        frame = cv2.dilate(frame, kernel, iterations=4)

        total_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(frame, connectivity=4)

        bbox_list = []

        for i in range(1, total_labels):
            x, y, w, h, area = stats[i]

            if (area > 2000):
                bbox = {
                "xtl": float(x),
                "ytl": float(y),
                "xbr": float(x + w),
                "ybr": float(y + h)
                }
                bbox_list.append(bbox)

                cv2.rectangle(color_frame[idx], (x, y), (x + w, y + h), (0, 255, 0), 2)

        out.write(color_frame[idx])
        
        if bbox_list:
            bbox_dict[idx+idx_frame] = bbox_list

    out.release()

    with open(f'bbox_task1_{alpha}.pkl', 'wb') as f:
        pickle.dump({'bboxes': bbox_dict}, f)
    
    return bbox_dict

def bbox2Coco(bboxes_dict, alpha, option, gt_json=False):

    if option == 'gt':
        coco_json = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "object"}]}

        annotation_id = 1

        for frame_id, bboxes in sorted(bboxes_dict.items()):
            coco_json["images"].append({
                "id": frame_id,
                "file_name": f"{frame_id}.jpg",
                "height": 1080,
                "width": 1920
            })

            for bbox in bboxes:
                x_min, y_min, x_max, y_max = bbox["xtl"], bbox["ytl"], bbox["xbr"], bbox["ybr"]
                width = x_max - x_min
                height = y_max - y_min

                coco_json["annotations"].append({
                    "image_id": frame_id,
                    "category_id": 1, 
                    "bbox": [x_min, y_min, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                    "id": annotation_id
                })
                annotation_id += 1

    elif option == 'predict':

        with redirect_stdout(io.StringIO()):
            gt = COCO(gt_json)

        coco_json = []

        for frame_id, bboxes in sorted(bboxes_dict.items()):
            if frame_id in gt.getImgIds():
                for bbox in bboxes:
                    x_min, y_min, x_max, y_max = bbox["xtl"], bbox["ytl"], bbox["xbr"], bbox["ybr"]
                    width, height = x_max - x_min, y_max - y_min

                    coco_json.append({
                        "image_id": frame_id,
                        "category_id": 1,
                        "bbox": [x_min, y_min, width, height],
                        "score": 0.5 # np.random.uniform(0, 1)
                    })
    else:
        pass

    with open(f'coco_{option}_{alpha}.json', "w") as f:
        json.dump(coco_json, f)
    
    return coco_json

def evaluate(frames, first_frame, alpha, K, gt_path, predict_path):
    with redirect_stdout(io.StringIO()):
        gt = COCO(gt_path)
        predict = gt.loadRes(predict_path)
        img_ids = set(gt.getImgIds())

        output_path = f'eva_task1_{alpha}.avi'
        fps = 20 
        frame_size = (frames.shape[2], frames.shape[1])

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)
        map_result = []
        map_result_auto = []
        for idx, frame in enumerate(frames):
            image_id = idx+first_frame
            mAPK = 0
            if image_id in img_ids:

                coco_eval = COCOeval(gt, predict, "bbox")
                coco_eval.params.imgIds = [image_id]
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                mAPK_auto = coco_eval.stats[1]
                 
                for img_eval in coco_eval.evalImgs:
                    if img_eval:
                        dt_matches = np.array(img_eval['dtMatches'][0])
                        tp_count = (dt_matches > 0).sum()
                        fp_count = (dt_matches == 0).sum() 
                        mAPK = tp_count / (tp_count + fp_count + 0.000001)
                
                map_result.append(mAPK)
                map_result_auto.append(mAPK_auto)

            gt_anns = gt.loadAnns(gt.getAnnIds(imgIds=image_id))
            pred_anns = predict.loadAnns(predict.getAnnIds(imgIds=image_id))

            #gt
            for ann in gt_anns:
                x, y, w, h = map(int, ann["bbox"])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "GT", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            #predict
            for ann in pred_anns:
                x, y, w, h = map(int, ann["bbox"])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Pred", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            #ap
            if image_id in img_ids:
                cv2.putText(frame, f"mAP@{K}: {mAPK:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            out.write(frame)

        out.release()

        coco_eval = COCOeval(gt, predict, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        coco_eval.stats[1]

    return coco_eval.stats[1], np.mean(map_result), np.mean(map_result_auto)