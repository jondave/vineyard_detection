import glob
from IPython.display import Image, display
import subprocess

# Define paths
weights_path = "/home/cabbage/Code/yolov9/runs/train/exp8/weights/best.pt"
source_path = "/home/cabbage/Code/yolov9/data/vineyard/test/images"
detect_script = "/home/cabbage/Code/yolov9/detect.py"
detected_images_path = "/home/cabbage/Code/yolov9/runs/detect/exp8/*.jpg"

# Run YOLO detection
detect_command = [
    "python", detect_script,
    "--img", "1280",
    "--conf", "0.1",
    "--device", "0",
    "--weights", weights_path,
    "--source", source_path
]

print(f"Running detection with the command: {' '.join(detect_command)}")
subprocess.run(detect_command)

# Display detected images
detected_images = glob.glob(detected_images_path)[:3]

if detected_images:
    for image_path in detected_images:
        display(Image(filename=image_path, width=600))
        print("\n")
else:
    print("No images found in the detection folder.")
