import json
import os
from ultralytics import YOLO
import cv2
from utils import frames2gif

def detect_cars_yolov8n(camera_sequence, video_path, output_folder):
    print("Detecting cars using YOLOv8n...")
    model_path = "yolov8n.pt"

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
    
    # Save predictions to a JSON file
    json_output_path = output_folder / f'predictions_yolov8n_{camera_sequence}.json'
    with open(json_output_path, 'w') as json_file:
        json.dump(predictions, json_file, indent=4)
    print(f"Predictions saved to {json_output_path}")

    # frames2gif(frames, output_folder / f'yolov8n_off_shelf_detection_{camera_sequence}.gif')

    cap.release()
    # out.release()
    cv2.destroyAllWindows()
    print(f"GIF with detected cars using YOLOv8 saved to {output_folder}")
    return predictions

    