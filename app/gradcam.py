import tensorflow as tf
import numpy as np
import cv2
import base64


LAST_CONV_LAYER = "conv5_block3_out"


def generate_gradcam(model, img_bgr):
    img = cv2.resize(img_bgr, (224, 224))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = np.expand_dims(img_rgb, axis=0)

    grad_model = tf.keras.models.Model(
        [model.input],
        [model.get_layer(LAST_CONV_LAYER).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]
    heatmap = conv_out @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-6

    heatmap = cv2.resize(heatmap.numpy(), (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    _, buffer = cv2.imencode(".png", overlay)
    return base64.b64encode(buffer).decode("utf-8")
