import streamlit as st
from PIL import Image
from helpers.ui import show_header, show_authors
from helpers.file_io import load_image
from helpers.face_module import recognize_faces
from helpers.hand_module import detect_hand_signal
from helpers.yolo_module import apply_yolo
from helpers.processing import apply_processing, functions_use_color
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase


# Cấu hình giao diện
logo = Image.open("logo-cntt.png")

st.set_page_config(layout="wide", page_title="Xử lý ảnh số")
show_header()
show_authors()

# Danh sách chức năng
chapters = {
    "YOLOv8": ["Nhận diện trái cây"],
    "Chapter 3 – Biến đổi độ sáng và lọc": functions_use_color + [
        "Negative", "Logarit", "Power", "Piecewise Line", "Histogram", "Hist Equal",
        "Local Hist", "Hist Stat", "Smooth Box", "Smooth Gauss", "Hubble", "Median Filter", "Sharp"
    ],
    "Chapter 4 – Xử lý miền tần số": ["Spectrum", "Remove Moire"],
    "Chapter 5 – Chuyển động": ["Create Motion", "DeMotion", "DeMotion Noise", "DeMotion Weiner"],
    "Chapter 9 – Hình thái học": ["Erosion", "Dilation", "Boundary", "Contour", "Convex Hull",
                                  "Defect Detect", "Hole Fill", "Connect Component", "Remove Small Rice"],
    "Face Detection": ["Nhận diện từ ảnh", "Nhận diện từ webcam"],
    "Hand Signal Detection": ["Nhận diện tay từ webcam"]
}

# Giao diện sidebar
st.sidebar.image(logo, width=200)
selected_chapter = st.sidebar.selectbox("📚 Chọn chương", list(chapters.keys()))
selected_function = st.sidebar.selectbox("🛠️ Chọn chức năng", chapters[selected_chapter])

uploaded_file = None
if selected_chapter not in ["Face Detection", "Hand Signal Detection"] or selected_function == "Nhận diện từ ảnh":
    uploaded_file = st.file_uploader("📂 Tải ảnh", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])

# Xử lý ảnh tĩnh
if uploaded_file:
    img_gray, img_color = load_image(uploaded_file)

    if selected_chapter == "YOLOv8":
        result_img = apply_yolo(img_color)
        input_img = img_color
    elif selected_chapter == "Face Detection":
        result_img = recognize_faces(img_color)
        input_img = img_color
    else:
        input_img = img_color if selected_function in functions_use_color else img_gray
        result_img = apply_processing(input_img, selected_function)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ảnh Gốc")
        st.image(input_img, channels="BGR" if len(input_img.shape) == 3 else "GRAY", use_column_width=True)
    with col2:
        st.subheader("Ảnh Sau Xử Lý")
        st.image(result_img, channels="BGR" if len(result_img.shape) == 3 else "GRAY", use_column_width=True)

# Xử lý webcam: Nhận diện khuôn mặt
elif selected_chapter == "Face Detection" and selected_function == "Nhận diện từ webcam":
    st.subheader("📷 Nhận diện khuôn mặt từ webcam")

    class FaceDetectionTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img = recognize_faces(img)
            return img

    webrtc_streamer(key="face-detection", video_transformer_factory=FaceDetectionTransformer)

# Xử lý webcam: Nhận diện tay
elif selected_chapter == "Hand Signal Detection" and selected_function == "Nhận diện tay từ webcam":
    st.subheader("✋ Nhận diện tay từ webcam")

    class HandDetectionTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img = detect_hand_signal(img)
            return img

    webrtc_streamer(key="hand-detection", video_transformer_factory=HandDetectionTransformer)
