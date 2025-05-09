from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__ + "/.."))
model_yolo = YOLO(os.path.join(BASE_DIR, "yolov8n_nhan_dien_trai_cay.onnx"), task="detect")

def apply_yolo(imgin):
    imgout = imgin.copy()
    annotator = Annotator(imgout)
    results = model_yolo.predict(imgin, conf=0.5, verbose=False)
    boxes = results[0].boxes.xyxy.cpu()
    clss = results[0].boxes.cls.cpu().tolist()
    confs = results[0].boxes.conf.tolist()
    names = model_yolo.names
    for box, cls, conf in zip(boxes, clss, confs):
        label = f"{names[int(cls)]} {conf:.2f}"
        annotator.box_label(box, label=label, txt_color=(255, 0, 0), color=(0, 255, 0))
    return imgout
