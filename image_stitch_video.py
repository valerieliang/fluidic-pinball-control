import cv2
import glob
import os

PNG_DIR = "snapshots_5000"
VIDEO_OUT = "pinball.mp4"
VIDEO_FPS = 30

images = sorted(glob.glob(os.path.join(PNG_DIR, "vort_*.png")))

if not images:
    raise RuntimeError("No PNG files found.")

# Read first frame to get size
frame = cv2.imread(images[0])
height, width, _ = frame.shape

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(VIDEO_OUT, fourcc, VIDEO_FPS, (width, height))

for img_path in images:
    frame = cv2.imread(img_path)
    video.write(frame)

video.release()
print(f"Video saved: {VIDEO_OUT}")