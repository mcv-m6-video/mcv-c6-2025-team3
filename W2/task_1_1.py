import os
from ultralytics import YOLO
import cv2

from utils import frames2gif

def get_yolov8_predictions(frame, model, conf_threshold=0.3):
    """
    Returns a list of tuples (x1, y1, x2, y2, class_id, confidence) for each detected object.
    """
    results = model(frame)  # Run the model on the frame
    boxes = []
    
    # Iterate over all detected results (boxes) in the current frame
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])  # Confidence score
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates (top-left x1, y1, bottom-right x2, y2)
            
            if conf >= conf_threshold:
                # Add the detection to the list if confidence is above the threshold
                boxes.append((x1, y1, x2, y2, cls_id, conf))
    
    return boxes

def detect_cars_yolov8n(video_path, output_folder):
    print("Detecting cars using YOLOv8n...")
    # Define the path to the model file
    model_path = "yolov8n.pt"

    # Check if the model file exists, and load it if not
    if not os.path.exists(model_path):
        model = YOLO(model_path)
    else:
        model = YOLO(model_path)

    # Define class names
    class_names = model.names
    # print(class_names) #class 2:'car'
    car_class_id = [k for k, v in class_names.items() if v == "car"]

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = output_folder / f'yolov8n_off_shelf_detection.avi'
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    # out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_number = 0 
    predictions = {}
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        # Perform inference on the current frame
        results = model(frame)

        predicted_boxes = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if cls_id in car_class_id and conf > 0.3:
                    # Draw bounding box and label on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Car: {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                # Add the bounding box to the list for this frame
                predicted_boxes.append((x1, y1, x2, y2, cls_id, conf))

        predictions[frame_number] = predicted_boxes
        # Write the processed frame to the output video
        frames.append(frame)
        # out.write(frame)

    frames2gif(frames, output_folder / f'yolov8n_off_shelf_detection.gif')

    cap.release()
    # out.release()
    cv2.destroyAllWindows()
    print(f"GIF with detected cars using YOLOv8 saved to {output_folder}")
    return predictions

    