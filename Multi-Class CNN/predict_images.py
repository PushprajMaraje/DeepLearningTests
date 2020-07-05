import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf 

MODEL_PATH = "G:\\natural\\model.h5"
IMG_PATH = "G:\\natural\\ComputerDesktopWallpapersCollection498_056.jpg"

model = tf.keras.models.load_model(MODEL_PATH)

img = image.load_img(IMG_PATH, target_size=(500,500))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

test_image = np.vstack([x])
classes = model.predict(test_image)
print("Predicted classes : ", classes)


