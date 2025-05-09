import cv2
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__ + "/.."))
MODEL_DIR = os.path.join(BASE_DIR, "model")

svc = joblib.load(os.path.join(MODEL_DIR, "svc.pkl"))
face_detector = cv2.FaceDetectorYN.create(
    os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx"), "", (320, 320), 0.9, 0.3, 5000
)
face_recognizer = cv2.FaceRecognizerSF.create(
    os.path.join(MODEL_DIR, "face_recognition_sface_2021dec.onnx"), ""
)

mydict = ['ChiThanh', 'NhuQuynh', 'ThanhDuy', 'VanPhat']

def recognize_faces(image):
    face_detector.setInputSize((image.shape[1], image.shape[0]))
    faces = face_detector.detect(image)
    if faces[1] is not None:
        for face in faces[1]:
            face_align = face_recognizer.alignCrop(image, face)
            face_feature = face_recognizer.feature(face_align)
            prediction = svc.predict(face_feature)
            name = mydict[prediction[0]]
            x, y, w, h = map(int, face[:4])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return image
