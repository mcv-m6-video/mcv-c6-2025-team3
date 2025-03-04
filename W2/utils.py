import os
from PIL import Image, ImageSequence
import cv2
import xml.etree.ElementTree as ET
import imageio

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

def frames2gif(frames_list, output_gif, fps=30):
   with imageio.get_writer(output_gif, mode='I', duration=1000 / fps) as writer:
        for frame in frames_list:
            frame_resized = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            writer.append_data(frame_rgb)

# def trim_gif(input_gif, output_gif, duration=10):
#     gif = Image.open(input_gif)

#     frame_rate = gif.info['duration']  
#     frames = []
    
#     for i, frame in enumerate(ImageSequence.Iterator(gif)):
#         if i * frame_rate / 1000 >= duration: 
#             break
#         frames.append(frame.copy())

#     frames[0].save(output_gif, save_all=True, append_images=frames[1:], duration=frame_rate, loop=0)
#     print(f"Trimmed GIF saved to {output_gif}")

def trim_gif(input_gif, output_gif, start_time=0, end_time=10):
    gif = Image.open(input_gif)

    frame_rate = gif.info['duration']
    
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

def create_gif(output_filename, start_frame, end_frame, folder):
    frames = []
    
    for i in range(start_frame, end_frame + 1):
        filename = os.path.join(folder, f"frame_{i}.png")
        if os.path.exists(filename):
            frames.append(Image.open(filename))
        else:
            print(f"Warning: {filename} not found. Skipping.")
    
    if frames:
        frames[0].save(output_filename, save_all=True, append_images=frames[1:], loop=0)
        print(f"GIF saved as {output_filename}")
    else:
        print("No frames found. GIF not created.")