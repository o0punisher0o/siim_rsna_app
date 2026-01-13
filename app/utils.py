import numpy as np
import cv2
from PIL import Image
import io


def read_image(file_bytes: bytes):
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
