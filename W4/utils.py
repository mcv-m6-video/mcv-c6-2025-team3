import json
import cv2
import imageio


def frames2gif(frames_list, output_gif, fps=30):
   with imageio.get_writer(output_gif, mode='I', duration=1000 / fps) as writer:
        for frame in frames_list:
            frame_resized = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            writer.append_data(frame_rgb)

def save_json(file: dict, name: str):
    with open(name, "w") as f:
        json.dump(file, f)