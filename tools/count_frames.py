import cv2

# Path to the video file
video_path = '/projects/kosecka/hongrui/dataset/how2sign/video_level/test/rgb_front/raw_videos/_fZbAxSSbX4-5-rgb_front.mp4'  # Adjust the extension if it's .avi or something else

# Given timestamps
start_time = 3.1  # in seconds
end_time = 6.91   # in seconds

start_timie = 7.25	
end_time = 9.25

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video file: {video_path}")

# Get FPS
fps = cap.get(cv2.CAP_PROP_FPS)

# Compute start and end frame numbers
start_frame = int(round(start_time * fps))
end_frame = int(round(end_time * fps))

# Total frames in the clip
num_frames = end_frame - start_frame + 1

print(f"FPS: {fps}")
print(f"Start Frame: {start_frame}")
print(f"End Frame: {end_frame}")
print(f"Number of frames from {start_time}s to {end_time}s: {num_frames}")

cap.release()
