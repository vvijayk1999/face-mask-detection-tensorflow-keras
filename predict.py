import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np


def prepare(filepath):
    IMG_SIZE = 100
    img = image.load_img(filepath, target_size=(100, 100))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    return np.vstack([x])

model = tf.keras.models.load_model('mask_detector_1594464012')

prediction = model.predict(prepare('1861.jpg'))
print(prediction)
prediction = model.predict(prepare('2083.jpg'))
print(prediction)

