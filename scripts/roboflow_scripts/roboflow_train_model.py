from ultralytics import YOLO
from matplotlib import pyplot as plt
from PIL import Image
from roboflow import Roboflow
import yaml

#Instance
# model = YOLO('yolov11n-seg.yaml')  # build a new model from YAML
model = YOLO('../../yolo_models/yolo11n-seg.pt')  # Transfer the weights from a pretrained model (recommended for training)

# rf = Roboflow(api_key="SY4FpqTfkiGFwCxMgZ5b")
# project = rf.workspace("vista-qsopb").project("vineyard_segmentation")
# version = project.version(1)
# dataset = version.download("yolov11")

# define number of classes based on YAML
with open("../../data/datasets/vineyard_segmentation-7/data.yaml", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])

#Define a project --> Destination directory for all results
project = "../../data/datasets/trained/vineyard_segmentation_v7/"
#Define subdirectory for this specific training
name = "run1" #note that if you run the training again, it creates a directory: 200_epochs-2

# Train the model
results = model.train(data='../../data/datasets/vineyard_segmentation-7/data.yaml',
                      project=project,
                      name=name,
                      epochs=100,
                      patience=0, #I am setting patience=0 to disable early stopping.
                      batch=16,
                      #imgsz=640,
                      device=0)