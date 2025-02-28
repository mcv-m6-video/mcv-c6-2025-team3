from PIL import Image, ImageSequence
import cv2
import imageio


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