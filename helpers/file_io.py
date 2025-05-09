import numpy as np
import cv2
from PIL import Image

def load_image(uploaded_file):
    pil_img = Image.open(uploaded_file)
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert("RGB")
    img_rgb = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return img_gray, img_bgr
