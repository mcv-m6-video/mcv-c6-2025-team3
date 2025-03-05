from PIL import Image, ImageSequence
import cv2
import xml.etree.ElementTree as ET
import imageio
import torch
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_iou
from pathlib import Path
from torch.utils.data import Dataset
import os 
import torchvision.transforms as T
import yaml
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush']

def read_annonations(annotations_path):
    "For each frame we will return a list of objects containing"
    "the bounding boxes present in that frame"
    "car_boxes[1417] to obtain all bounding boxes in frame 1417"
    print('Reading annotations...')
    # Parse the XML file
    tree = ET.parse(annotations_path)
    root = tree.getroot()

    class_map = {"car": 3, "bike": 2}

    targets = []
    frame_dict = {}

    # Iterate over all <track> elements
    for track in root.findall('.//track'):
        label = track.get("label")      

        if label not in class_map or label == "bike":
            continue

        # Iterate over all <box> elements within each track
        for box in track.findall('box'):
            
            # Extract frame and bounding box coordinates
            frame = int(box.get('frame'))
            x_min = float(box.get('xtl'))
            y_min = float(box.get('ytl'))
            x_max = float(box.get('xbr'))
            y_max = float(box.get('ybr'))
            box_attributes = [x_min, y_min, x_max, y_max]    

            class_id = class_map[label]
            
            if frame not in frame_dict:
                frame_dict[frame] = {"boxes": [], "labels": []}

            frame_dict[frame]["boxes"].append(box_attributes)
            frame_dict[frame]["labels"].append(class_id)

    targets = [
        {"boxes": torch.tensor(frame_dict[frame]["boxes"]), "labels": torch.tensor(frame_dict[frame]["labels"])}
        for frame in sorted(frame_dict.keys())
    ]

    return targets, sorted(frame_dict.keys())

def frames2gif(frames_list, output_gif, fps=30):
   with imageio.get_writer(output_gif, mode='I', duration=1000 / fps) as writer:
        for frame in frames_list:
            frame_resized = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            writer.append_data(frame_rgb)


def trim_gif(input_gif, output_gif, start_time=0, end_time=10):
    gif = Image.open(input_gif)

    frame_rate = gif.info['duration'] if gif.info['duration'] > 100 else 150
    
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

def compute_mIoU(predictions, gt):
    iou_values = []

    for pred, g in zip(predictions, gt):
        if len(pred["boxes"]) > 0 and len(g["boxes"]) > 0:
            ious = box_iou(pred["boxes"], g["boxes"])
            max_ious = ious.max(dim=1)[0]
            iou_values.extend(max_ious.tolist())
        else:
            iou_values.append(0)
    return sum(iou_values) / len(iou_values) if len(iou_values) > 0 else 0

def evaluate(gt, predictions, output_path):
    metric_map = MeanAveragePrecision()
    metric_map.update(predictions, gt)
    map_result = metric_map.compute()

    miou = compute_mIoU(predictions, gt)
    
    with open(output_path, "w") as f:
        for key, value in map_result.items():
            v = value.tolist()

            if isinstance(v, list):  
                f.write(f"{key}: {v}\n")
            else:
                f.write(f"{key}: {v:.4f}\n")
        f.write(f"mIoU: {miou:.4f}\n")

def video2frames(video_path):
    output_folder = Path('/home/danielpardo/c6/frames/')
    output_folder.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_number = 0 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_filename = f"{output_folder}/frame_{frame_number:04d}.jpg"
        cv2.imwrite(frame_filename, frame)
        frame_number += 1
    cap.release()

class CarDataset(Dataset):
    def __init__(self, image_paths, annotations):
        #images path
        self.image_paths = image_paths

        #annotations
        self.targets = annotations

        self.transform = T.Compose([T.ToTensor()])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        #read img
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)

        #annotation
        target = self.targets[idx]

        return img, target

def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def get_dataloader(image_paths, gt_annotations, batch_size, strategy):

    if strategy == 'a':
        split_idx = int(len(image_paths) * 0.25)  # first 25% train

        train_p, test_p = image_paths[:split_idx], image_paths[split_idx:]
        train_a, test_a = gt_annotations[:split_idx], gt_annotations[split_idx:]

        train_dataset = CarDataset(train_p, train_a)
        test_dataset = CarDataset(test_p, test_a)

        train_dataloader = [DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)] #shuffle = false for sequence also in train
        test_dataloader = [DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)]
    elif strategy in ['b', 'c']:
        if strategy == 'c':
            image_paths, gt_annotations = shuffle(image_paths, gt_annotations, random_state=42)

        kf_seq = KFold(n_splits=4, shuffle=(strategy == 'c'), random_state=None if (strategy == 'b') else 42) #shuffle = false for sequence // rs None if shuffle == false, o/w 42

        train_dataloader = []
        test_dataloader = []
        for fold, (train_idx, test_idx) in enumerate(kf_seq.split(image_paths)):
            train_seq_paths = [image_paths[i] for i in train_idx]
            test_seq_paths = [image_paths[i] for i in test_idx]
            print(train_idx)
            train_seq_annotations = [gt_annotations[i] for i in train_idx]
            test_seq_annotations = [gt_annotations[i] for i in test_idx]

            train_seq_dataset = CarDataset(train_seq_paths, train_seq_annotations)
            test_seq_dataset = CarDataset(test_seq_paths, test_seq_annotations)

            train_dataloader.append(DataLoader(train_seq_dataset, batch_size=batch_size, shuffle=(strategy == 'c'), collate_fn=collate_fn)) #shuffle = false for sequence also in train
            test_dataloader.append(DataLoader(test_seq_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn))

    return train_dataloader, test_dataloader


def annotations2yolo(image_paths, annotations, output_folder):

    output_img = Path(f'{output_folder}/images/')
    output_img.mkdir(exist_ok=True)

    output_labels = Path(f'{output_folder}/labels/')
    output_labels.mkdir(exist_ok=True)


    for img_path, anno in zip(image_paths, annotations):

        img = cv2.imread(img_path)
        h, w, _ = img.shape #image size
        
        #set filenaem
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = f"{output_labels}/{base_name}.txt"

        cv2.imwrite(f'{output_img}/{base_name}.jpg', img)

        with open(txt_path, "w") as f:
            for box, label in zip(anno['boxes'], anno['labels']):
                xmin, ymin, xmax, ymax = box.numpy()
                #compute bbox in yolo format
                x_center = ((xmin + xmax) / 2.0) / w
                y_center = ((ymin + ymax) / 2.0) / h
                width    = (xmax - xmin) / w
                height   = (ymax - ymin) / h

                class_id = 0 #only 1 class = 'car'

                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")



def data2yolo(image_paths, gt_annotations, yolo_folder):
    output_folder = Path(yolo_folder)
    output_folder.mkdir(exist_ok=True)

    for strategy in ['a', 'b', 'c']:
        if strategy == 'a':
            split_idx = int(len(image_paths) * 0.25)  # first 25% train

            train_p, test_p = image_paths[:split_idx], image_paths[split_idx:]
            train_a, test_a = gt_annotations[:split_idx], gt_annotations[split_idx:]
            
            output_train = Path(f'{output_folder}/{strategy}/train/')
            output_train.mkdir(parents=True, exist_ok=True)

            output_val = Path(f'{output_folder}/{strategy}/val/')
            output_val.mkdir(parents=True, exist_ok=True)

            annotations2yolo(train_p, train_a, output_train)
            annotations2yolo(test_p, test_a, output_val)
            data_content = {
            "train": f"{output_train}/images",
            "val": f"{output_val}/images",
            "names": ["car"] 
            }

            with open(f'{output_folder}/data_{strategy}.yaml', "w") as f:
                yaml.dump(data_content, f, sort_keys=False)

        elif strategy in ['b', 'c']:
                if strategy == 'c':
                    image_paths, gt_annotations = shuffle(image_paths, gt_annotations, random_state=42)
                kf_seq = KFold(n_splits=4, shuffle=(strategy == 'c'), random_state=None if (strategy == 'b') else 42) #shuffle = false for sequence // rs None if shuffle == false, o/w 42


                for fold, (train_idx, test_idx) in enumerate(kf_seq.split(image_paths)):
                    train_seq_paths = [image_paths[i] for i in train_idx]
                    test_seq_paths = [image_paths[i] for i in test_idx]

                    train_seq_annotations = [gt_annotations[i] for i in train_idx]
                    test_seq_annotations = [gt_annotations[i] for i in test_idx]

                    output_train = Path(f'{output_folder}/{strategy}/{fold}/train/')
                    output_train.mkdir(parents=True, exist_ok=True)

                    output_val = Path(f'{output_folder}/{strategy}/{fold}/val/')
                    output_val.mkdir(parents=True, exist_ok=True)

                    annotations2yolo(train_seq_paths, train_seq_annotations, output_train)
                    annotations2yolo(test_seq_paths, test_seq_annotations, output_val)

                    data_content = {
                    "train": f"{output_train}/images",
                    "val": f"{output_val}/images",
                    "names": ["car"] 
                    }

                    with open(f'{output_folder}/data_{strategy}_{fold}.yaml', "w") as f:
                        yaml.dump(data_content, f, sort_keys=False)


