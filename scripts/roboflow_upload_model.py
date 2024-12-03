import roboflow
import json

import sys
sys.path.append('/home/cabbage/Code/yolov9')

# Load the API key
with open('../config/api_key.json', 'r') as file:
    config = json.load(file)
ROBOFLOW_API_KEY = config.get("ROBOFLOW_API_KEY")    

rf = roboflow.Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace().project("vineyard_test")

#can specify weights_filename, default is "weights/best.pt"
version = project.version(5)
version.deploy("yolov9", "/home/cabbage/Code/yolov9/runs/train/exp14", "weights/best.pt")