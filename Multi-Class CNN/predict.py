import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf 
model = tf.keras.models.load_model('G:\\natural\\model.h5')

path = 'G:\\natural\\ComputerDesktopWallpapersCollection498_056.jpg' 
img = image.load_img(path, target_size=(500,500))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

imagea = np.vstack([x])
classes = model.predict(imagea)
print(classes)


