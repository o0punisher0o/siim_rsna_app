import tensorflow as tf
import numpy as np
import cv2

from tensorflow.keras.applications.resnet50 import preprocess_input


CLASSES = ["negative", "typical", "indeterminate", "atypical"]
IMG_SIZE = (224, 224)

MODEL_PATH = "models/resnet50_siim_cw_es_best.h5"

model = tf.keras.models.load_model(MODEL_PATH)


def predict_image(img_bgr: np.ndarray):
    img = cv2.resize(img_bgr, IMG_SIZE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    x = np.expand_dims(img_rgb, axis=0)
    x = preprocess_input(x)

    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))

    return {
        "predicted_class": CLASSES[idx],
        "confidence": float(probs[idx]),
        "probs": {CLASSES[i]: float(p) for i, p in enumerate(probs)}
    }
