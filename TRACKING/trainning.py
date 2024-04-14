import os
from IPython import display
display.clear_output()
import ultralytics
ultralytics.checks()
import glob
from ultralytics import YOLO

from roboflow import Roboflow
rf = Roboflow(api_key="rY2qwZpfBG3KU1GN1xc5")
project = rf.workspace("phanquangduy248").project("lab-mouse")
version = project.version(2)
dataset = version.download("yolov8")


#CLI:

#chạy lệnh sau để training:     yolo task=segment mode=train model=yolov8n-seg.pt data=D:\NCKH2023-2024\Lab-mouse-2\data.yaml epochs=50 imgsz=640
#                 validate:     yolo task=segment mode=val model={HOME}/runs/segment/train/weights/best.pt data=D:\NCKH2023-2024\Lab-mouse-2\data.yaml
#                 inference:    yolo task=segment mode=predict model=runs/segment/train17/weights/best.pt conf=0.25 source=D:\NCKH2023-2024\Lab-mouse-2\test\images save=true


# for image_path in glob.glob(f'runs/segment/predict/*.jpg')[:3]:
#       display(Image(filename=image_path, height=600))
#       print("\n")