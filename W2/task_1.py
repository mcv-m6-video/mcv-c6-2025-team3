from ultralytics import YOLO
import cv2
import torch
from utils_1 import frames2gif, COCO_INSTANCE_CATEGORY_NAMES, evaluate, trim_gif
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import (fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights)
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pathlib import Path
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_yolo(model, frame, conf_th, byCar):

    results = model(frame, verbose=False)

    predictions = []
    
    for result in results:
        boxes = []
        scores = []
        labels = []

        for box in result.boxes:
            cls_id = int(box.cls[0])+1  #id
            conf = float(box.conf[0])  #score
            x1, y1, x2, y2 = map(float, box.xyxy[0])  #bbox
            
            class_name = COCO_INSTANCE_CATEGORY_NAMES[cls_id]

            if (byCar and class_name == 'car' and conf >= conf_th) or (not byCar and conf >= conf_th):
                boxes.append([x1, y1, x2, y2])
                scores.append(conf)
                labels.append(cls_id)

                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{class_name}: {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        predictions.append({
            "boxes": torch.tensor(boxes),
            "scores": torch.tensor(scores),
            "labels": torch.tensor(labels)
        })

    return predictions, frame


def predict_fasterrcnn(model, frame, conf_th, byCar, transform, finetune = False):    
    if not isinstance(frame, torch.Tensor):
        img_pil = Image.fromarray(frame)
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
    else:
        img_tensor = frame.squeeze(0)

    img_tensor = img_tensor.squeeze(0)

    with torch.no_grad():
        outputs = model([img_tensor])

    pred_boxes = outputs[0]['boxes'].cpu().numpy()
    pred_scores = outputs[0]['scores'].cpu().numpy()
    pred_labels = outputs[0]['labels'].cpu().numpy()

    predictions = []
    boxes = []
    scores = []
    labels = []

    for i in range(len(pred_boxes)):
        conf = pred_scores[i]
        cls_id = pred_labels[i]

        if cls_id > 80:
            continue
        class_name = COCO_INSTANCE_CATEGORY_NAMES[cls_id]

        if finetune and cls_id == 1:
            class_name = 'car'
            #cls_id = 3

        if (byCar and class_name == 'car' and conf >= conf_th) or (not byCar and conf >= conf_th):
            x1, y1, x2, y2 = map(int, pred_boxes[i])
            boxes.append([x1, y1, x2, y2])
            scores.append(conf)
            labels.append(cls_id)
            if isinstance(frame, torch.Tensor):
                frame = cv2.cvtColor((frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{class_name}: {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
    predictions.append({
        "boxes": torch.tensor(boxes),
        "scores": torch.tensor(scores),
        "labels": torch.tensor(labels)
    })

    return predictions, frame



def predict_video(option, video_path, gt, frames_gt, model, conf_th, output_folder, byCar, gif, transform = None):

    output_path = f'{output_folder}/{option}_{conf_th}_{byCar}'

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_number = 0 
    predictions = []
    frames = []
    frames_pre_gt = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number in frames_gt:
            # Perform inference on the current frame
            if option == 'yolov8n':
                prediction, frame = predict_yolo(model, frame, conf_th, byCar)
            elif option == 'fasterrcnn':
                prediction, frame = predict_fasterrcnn(model, frame, conf_th, byCar, transform)

            frame_pre_gt = frame.copy()

            for idx, box in enumerate(gt[frame_number]['boxes']):
                cls_id = int(gt[frame_number]['labels'][idx])  #id
                x1, y1, x2, y2 = map(int, box)  #bbox
                class_name = COCO_INSTANCE_CATEGORY_NAMES[cls_id]

                if byCar:
                        cv2.rectangle(frame_pre_gt, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame_pre_gt, f"{class_name}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                else:
                        cv2.rectangle(frame_pre_gt, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame_pre_gt, f"{class_name}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            predictions.append(prediction[0])
            frames.append(frame)
            frames_pre_gt.append(frame_pre_gt)

        frame_number += 1

    cap.release()
    if gif:
        frames2gif(frames, f'{output_path}_predict.gif')
        frames2gif(frames_pre_gt, f'{output_path}_predict_gt.gif')

    return predictions

def detect_cars_yolov8n(video_path, gt, frames_gt, conf_th, output_folder, byCar, gif):
    print("Detecting cars using YOLOv8n...")
    # Define the path to the model file
    model_path = "yolov8n.pt"
    model = YOLO(model_path)
    predictions = predict_video('yolov8n', video_path, gt, frames_gt, model, conf_th, output_folder, byCar, gif)
    return predictions

def detect_cars_fasterrcnn(video_path, gt, frames_gt, conf_th, output_folder, byCar, gif):
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    transform = weights.transforms()
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    model.eval()
    model = model.to(device)
    predictions = predict_video('fasterrcnn', video_path, gt, frames_gt, model, conf_th, output_folder, byCar, gif, transform)
    return predictions


def finetuneYolo(input_folder, output_folder, strategy, batch_size, epochs):
    if strategy == 'a':
        model = YOLO("yolov8n.pt")
        model.train(
            data=f"{input_folder}/data_{strategy}.yaml",
            epochs=epochs,
            batch=batch_size,
            freeze=10,
            project=f'{output_folder}',
            name=f"{strategy}"
        )
    else:
        for fold in range(4):
            model = YOLO("yolov8n.pt")
            
            data_fold_path = f"{input_folder}/data_{strategy}_{fold}.yaml"

            model.train(
                data=data_fold_path,
                epochs=epochs,
                batch=batch_size,
                freeze=10,
                project=f'{output_folder}',
                name=f"{strategy}_{fold}"
            )


def finetuneFasterrcnn(dataloaders, output_folder, strategy, epochs, conf_th):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 2  # car + background

    train_dataloaders, test_dataloaders = dataloaders

    for fold_idx, (train_dataloader, test_dataloader) in enumerate(zip(train_dataloaders, test_dataloaders)):
        if strategy == 'a':
            output_path = Path(f'{output_folder}/{strategy}')
            output_path.mkdir(exist_ok=True)
        else:
            output_path = Path(f'{output_folder}/{strategy}_{fold_idx}')
            output_path.mkdir(exist_ok=True)
        
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        transform = weights.transforms()
        model = fasterrcnn_resnet50_fpn_v2(weights=weights)

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)  #modify hdead


        #freeze all layers
        for param in model.backbone.parameters():
            param.requires_grad = False
        #unfreeze last layer to finetuning
        for param in model.backbone.body.layer4.parameters():
            param.requires_grad = True

        model.to(device)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=0.0001)

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0

            for images, targets in train_dataloader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) if k != "labels" else v.long().to(device) for k, v in t.items()} for t in targets]
                for t in targets:
                    t["boxes"] = t["boxes"].to(device)
                    t["labels"] = torch.ones_like(t["labels"], dtype=torch.long, device=device)

                optimizer.zero_grad()
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                losses.backward()
                optimizer.step()
                epoch_loss += losses.item()

                del images, targets, loss_dict, losses
                torch.cuda.empty_cache()
                break

        model.eval()
        

        with torch.no_grad():

            predictions = []
            gt_list = []
            frames = []
            frames_pre_gt = []

            for images, targets in test_dataloader:

                images = [img.to(device) for img in images]
                for t in targets:
                    t["boxes"] = t["boxes"].to(device)
                    t["labels"] = torch.ones_like(t["labels"], dtype=torch.long, device=device)


                for img, gt in zip(images, targets):
                    prediction, frame = predict_fasterrcnn(model, img, conf_th, True, transform, True)

                    if isinstance(frame, torch.Tensor):
                        frame_pre_gt = cv2.cvtColor((frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    else:
                        frame_pre_gt = frame.copy()

                    for idx, box in enumerate(gt['boxes']):
                        cls_id = int(gt['labels'][idx])  #id
                        x1, y1, x2, y2 = map(int, box)  #bbox
                        class_name = COCO_INSTANCE_CATEGORY_NAMES[cls_id]

                        cv2.rectangle(frame_pre_gt, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame_pre_gt, f"{class_name}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    predictions.append(prediction[0])
                    gt_list.append(gt)
                    frames.append(frame)
                    frames_pre_gt.append(frame_pre_gt)

                del images, targets
                torch.cuda.empty_cache()

            gt_list = [{k: v.cpu() for k, v in target.items()} for target in gt_list]
            predictions = [{k: v.cpu() for k, v in pred.items()} for pred in predictions]
            evaluate(gt_list, predictions, f'{output_path}/results_fasterrcnn_{conf_th}.txt')
            frames2gif(frames, f'{output_path}/fasterrcnn_predict.gif')
            frames2gif(frames_pre_gt, f'{output_path}/fasterrcnn_predict_gt.gif')
            
            cut_gif = f'{output_path}/fasterrcnn_predict.gif'
            trim_gif(cut_gif, f'{output_path}/fasterrcnn_predict_trimmed.gif', start_time=0, end_time=20)
            cut_gif = f'{output_path}/fasterrcnn_predict_gt.gif'
            trim_gif(cut_gif, f'{output_path}/fasterrcnn_predict_gt_trimmed.gif', start_time=0, end_time=20)

