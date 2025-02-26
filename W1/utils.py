import cv2
import xml.etree.ElementTree as ET
import numpy as np
import imageio
import numpy as np
import cv2
import pickle
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import io
from contextlib import redirect_stdout


def read_video(video_path):
    """
    Reads a video and returns frames in both color and grayscale.

    Args:
        video_path (str): Path to the video file.

    Returns:
        color_frames: np.ndarray of shape [n_frames, height, width, 3]
        gray_frames: np.ndarray of shape [n_frames, height, width]
    """
    print('Reading video...')
    # Open the video
    video = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not video.isOpened():
        print("Error: Cannot open video file.")
        exit()

    color_frames = []
    gray_frames = []
    
    # Read frames until the end
    while True:
        ret, frame = video.read()
        if not ret:
            print("End of video or cannot read frame.")
            break

        color_frames.append(frame)
        gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    # Release video capture
    video.release()

    # Return frames as NumPy arrays
    return np.array(color_frames), np.array(gray_frames)

def split_video_25_75(video_frames):
    """
    We want to return video frames separated in 25% and 75%

    Returns:
        frames_25: np.ndarray([n_frames_25, 1080, 1920, 3])
        frames_75: np.ndarray([n_frames_75, 1080, 1920, 3])
    """
    print('Splitting video in 25% and 75%...')
    split = int(video_frames.shape[0] * 0.25)
    return video_frames[:split], video_frames[split:]

def read_annonations(annotations_path):
    "For each frame we will return a list of objects containing"
    "the bounding boxes present in that frame"
    "car_boxes[1417] to obtain all bounding boxes in frame 1417"
    print('Reading annotations...')
    # Parse the XML file
    tree = ET.parse(annotations_path)
    root = tree.getroot()

    car_boxes = {}  # Store extracted boxes here

    # Iterate over all <track> elements
    for track in root.findall('.//track[@label="car"]'):
        # Iterate over all <box> elements within each track
        for box in track.findall('box'):
            # Check if the <attribute> name is not 'parked'
            parked_attribute = box.find('attribute[@name="parked"]')
            if parked_attribute is not None and parked_attribute.text == 'true':
                continue
            # Extract frame and bounding box coordinates
            frame = int(box.get('frame'))
            box_attributes = {
                'xtl': float(box.get('xtl')),
                'ytl': float(box.get('ytl')),
                'xbr': float(box.get('xbr')),
                'ybr': float(box.get('ybr')),
            }                 
            
            if frame in car_boxes:
                car_boxes[frame].append(box_attributes)
            else:
                car_boxes[frame] = [box_attributes]

    return car_boxes

def get_predicted_bounding_boxes(frame):
    """
    Extract bounding boxes from a single frame using Connected Components.
    
    Args:
        frame (np.ndarray): Binary frame (bool or uint8) [H, W].
        
    Returns:
        List[dict]: List of bounding boxes with keys: xtl, ytl, xbr, ybr.
    """
    # Ensure frame is uint8 (0 or 255)
    frame_uint8 = frame.astype(np.uint8) * 255

    # Connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(frame_uint8, connectivity=8)

    boxes = []
    for i in range(1, num_labels):  # Skip the background (label 0)
        x, y, w, h, area = stats[i]
        boxes.append({'xtl': x, 'ytl': y, 'xbr': x + w, 'ybr': y + h})
    return boxes

def visualize_foreground(foreground_segmented, color_frames_75, wait_time=30):
    """
    Visualize the segmented foreground using OpenCV.

    Args:
        foreground_segmented (np.ndarray): Boolean array of foreground pixels [n_frames, height, width].
        color_frames_75 (np.ndarray): Color frames corresponding to foreground segmentation.
        wait_time (int): Delay between frames in milliseconds (default: 30).

    Returns:
        None
    """
    for i in range(foreground_segmented.shape[0]):
        # Get current frame
        frame = color_frames_75[i].copy()

        # Highlight foreground in red (for visualization)
        frame[foreground_segmented[i]] = [0, 0, 255]  # Red color in BGR

        # Display the frame
        cv2.imshow("Foreground Segmentation", frame)

        # Wait for the specified time or until 'q' is pressed
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    # Close the window after visualization
    cv2.destroyAllWindows()

def save_foreground(foreground_segmented, color_frames_75, alpha, output_folder, rho= None, fps=30):
    if rho is None:
        output_path = output_folder / f'foreground_task1_{alpha}.avi'
    else:
        output_path = output_folder / f'foreground_task2_{alpha}_{rho}.avi'

    fps = 30 
    frame_size = (color_frames_75.shape[2], color_frames_75.shape[1])
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)
    frames = []
    for i in range(foreground_segmented.shape[0]):
        frame = color_frames_75[i].copy()

        frame[foreground_segmented[i]] = [0, 0, 255]

        out.write(frame)
        frames.append(frame)
    out.release()
    if rho is None:
        frames2gif(frames, output_folder / f'foreground_task1_{alpha}.gif')
    else:
        frames2gif(frames, output_folder / f'foreground_task2_{alpha}_{rho}.gif')


def frames2gif(frames_list, output_gif, fps=30):
   with imageio.get_writer(output_gif, mode='I', fps=fps) as writer:
        for frame in frames_list:
            frame_resized = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            writer.append_data(frame_rgb)

def compute_bbox(foreground_segmented, frames, alpha, idx_frame, output_folder, rho = None):

    foreground_gray = (foreground_segmented * 255).astype(np.uint8)
    color_frame = frames.copy()
    if rho is None:
        output_path = output_folder / f'bbox_task1_{alpha}.avi'
    else:
        output_path = output_folder / f'bbox_task2_{alpha}_{rho}.avi'

    fps = 30 
    frame_size = (color_frame.shape[2], color_frame.shape[1])

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

    bbox_dict = {}

    for idx, gray_frame in enumerate(foreground_gray):
        frame = cv2.medianBlur(gray_frame, 5)
        frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        kernel = np.ones((3,3), np.uint8)
        frame = cv2.dilate(frame, kernel, iterations=6)
        frame = cv2.erode(frame, kernel, iterations=4)

        total_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(frame, connectivity=4)

        bbox_list = []

        for i in range(1, total_labels):
            x, y, w, h, area = stats[i]
            bbox_area = w * h
            if bbox_area > 0 and (area / bbox_area) > 0.5 and area > 1000:
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
    if rho is None:
        with open(output_folder / f'bbox_task1_{alpha}.pkl', 'wb') as f:
            pickle.dump({'bboxes': bbox_dict}, f)
    else:
        with open(output_folder / f'bbox_task2_{alpha}_{rho}.pkl', 'wb') as f:
            pickle.dump({'bboxes': bbox_dict}, f)

    return bbox_dict

def bbox2Coco(bboxes_dict, alpha, option, output_folder, gt_json=False):

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
                    "image_id": int(frame_id),
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
                        "image_id": int(frame_id),
                        "category_id": 1,
                        "bbox": [x_min, y_min, width, height],
                        "score": 0.5 # np.random.uniform(0, 1)
                    })
    else:
        pass

    with open(output_folder / f'coco_{option}_{alpha}.json', "w") as f:
        json.dump(coco_json, f)
    
    return coco_json

def evaluate(frames, first_frame, alpha, K, gt_path, predict_path, output_folder, rho = None):

    with redirect_stdout(io.StringIO()):
        gt = COCO(gt_path)
        predict = gt.loadRes(predict_path)
        img_ids = set(gt.getImgIds())

        if rho is None:
            output_path = output_folder / f'eva_task1_{alpha}.avi'
        else:
            output_path = output_folder / f'eva_task2_{alpha}_{rho}.avi'

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

    if rho is None:
        frames2gif(frames, output_folder / f'eva_task1_{alpha}.gif')

    else:
        frames2gif(frames, output_folder / f'eva_task2_{alpha}_{rho}.gif')

    return coco_eval.stats[1], np.mean(map_result), np.mean(map_result_auto)