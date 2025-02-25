import cv2
import xml.etree.ElementTree as ET
import os
import numpy as np

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

def save_foreground(foreground_segmented, color_frames_75, alpha, fps=30):
    output_path = f'foreground_task1_{alpha}.avi'
    fps = 30 
    frame_size = (color_frames_75.shape[2], color_frames_75.shape[1])

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

    for i in range(foreground_segmented.shape[0]):
        frame = color_frames_75[i].copy()

        frame[foreground_segmented[i]] = [0, 0, 255]

        out.write(frame)
    out.release()