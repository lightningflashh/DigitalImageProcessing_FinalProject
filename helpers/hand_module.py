import onnxruntime
import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__ + "/.."))
hand_session = onnxruntime.InferenceSession(
    os.path.join(BASE_DIR, "handsignal_yolov8.onnx"), providers=["CPUExecutionProvider"]
)
hand_input_name = hand_session.get_inputs()[0].name
hand_output_name = hand_session.get_outputs()[0].name
hand_class_names = ['Power', 'Rock', 'Thumb Down', 'Thumb Up', 'Victory', 'Hi-Five']

def detect_hand_signal(frame):
    input_image = cv2.resize(frame, (640, 640))
    input_tensor = input_image[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)

    outputs = hand_session.run([hand_output_name], {hand_input_name: input_tensor})[0]
    for pred in outputs[0]:
        x1, y1, x2, y2, conf, cls = pred[:6]
        if conf > 0.7:
            class_id = int(cls)
            label = f"{hand_class_names[class_id]} {conf:.2f}" if 0 <= class_id < len(hand_class_names) else f"Unknown {conf:.2f}"
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
    return frame
