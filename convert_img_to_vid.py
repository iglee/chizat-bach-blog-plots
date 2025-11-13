import cv2
import glob
import re

def numerical_sort(value):
    # Extract the number at the end of the filename
    numbers = re.findall(r'\d+', value)
    return int(numbers[-1]) if numbers else -1

# Get all matching PNG files and sort them numerically
images = sorted(glob.glob("dynamics_wd_*.png"), key=numerical_sort)

if not images:
    raise ValueError("No images found with pattern dynamics_wd_*.png")

# Read the first image to get dimensions
frame = cv2.imread(images[0])
height, width, _ = frame.shape

# Define video writer (codec mp4v, 30 FPS)
out = cv2.VideoWriter('output_short.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

for img_name in images:
    img = cv2.imread(img_name)
    if img is None:
        print(f"Warning: Could not read {img_name}, skipping.")
        continue
    out.write(img)

out.release()
print("✅ Video saved as output_short.mp4")
