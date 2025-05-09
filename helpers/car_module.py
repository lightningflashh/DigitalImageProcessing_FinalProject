import os
import cv2
import math
import time
import tempfile
from ultralytics import YOLO

# === ĐƯỜNG DẪN MODEL
BASE_DIR = os.path.dirname(os.path.abspath(__file__ + "/.."))
MODEL_DIR = os.path.join(BASE_DIR, "model")
model_car = YOLO(os.path.join(MODEL_DIR, "car_detect.pt"))

def run_vehicle_detection(video_file, stframe=None):
    input_w, input_h = 460, 360
    laser_line = input_h - 120
    max_distance = 80

    def get_box_info(box):
        (x, y, w, h) = [int(v) for v in box]
        center_X = int((x + x + w) / 2)
        center_Y = int((y + y + h) / 2)
        return x, y, w, h, center_X, center_Y

    def is_old(center_Xd, center_Yd, boxes):
        for box_tracker in boxes:
            (xt, yt, wt, ht) = [int(c) for c in box_tracker]
            center_Xt = int((xt + xt + wt) / 2)
            center_Yt = int((yt + yt + ht) / 2)
            distance = math.sqrt((center_Xt - center_Xd)**2 + (center_Yt - center_Yd)**2)
            if distance < max_distance:
                return True
        return False

    def get_object_yolo(frame, conf_thres=0.3):
        results = model_car.predict(source=frame, conf=conf_thres, verbose=False)[0]
        boxes = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            if class_id == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                boxes.append([x1, y1, x2 - x1, y2 - y1])
        return boxes

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    vid = cv2.VideoCapture(tfile.name)
    out = cv2.VideoWriter("output_yolo_track.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (input_w, input_h))

    frame_count = 0
    car_number = 0
    obj_cnt = 0
    curr_trackers = []

    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break

        frame = cv2.resize(frame, (input_w, input_h))
        boxes = []
        old_trackers = curr_trackers
        curr_trackers = []

        for car in old_trackers:
            tracker = car['tracker']
            success, box = tracker.update(frame)
            if not success:
                continue
            boxes.append(box)
            x, y, w, h, center_X, center_Y = get_box_info(box)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if center_Y > laser_line and not car.get("counted", False):
                car_number += 1
                car["counted"] = True
            curr_trackers.append(car)

        if frame_count % 6 == 0:
            boxes_d = get_object_yolo(frame)
            for box in boxes_d:
                xd, yd, wd, hd, center_Xd, center_Yd = get_box_info(box)
                if not is_old(center_Xd, center_Yd, boxes):
                    tracker = cv2.legacy.TrackerMOSSE_create()
                    tracker.init(frame, tuple(box))
                    counted = center_Yd > laser_line - 10
                    if counted:
                        car_number += 1
                    curr_trackers.append({'tracker_id': obj_cnt, 'tracker': tracker, 'counted': counted})
                    obj_cnt += 1

        frame_count += 1
        cv2.putText(frame, f"Car number: {car_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.line(frame, (0, laser_line), (input_w, laser_line), (0, 0, 255), 2)
        out.write(frame)

        if stframe:
            stframe.image(frame, channels="BGR")

    vid.release()
    out.release()
    time.sleep(1)