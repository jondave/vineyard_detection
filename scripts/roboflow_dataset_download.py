from roboflow import Roboflow
rf = Roboflow(api_key="API_KEY_HERE")
project = rf.workspace("gaia-hse8w").project("semanticblt")
version = project.version(1)
dataset = version.download("yolov9")
