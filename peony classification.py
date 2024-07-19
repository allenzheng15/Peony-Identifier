import tensorflow as tf
import os 
import numpy as np 
import cv2
import imghdr
from tensorflow.keras.models import load_model 

model = load_model(os.path.join('models', 'peonymodelv4.h5'))

image = cv2.imread('daisytest2.jpg')
resized = tf.image.resize(image, (256,256))
value = model.predict(np.expand_dims(resized/255, 0))

print(value)
if value < 0.5:
    print("It is not a peony")
else:
    print("It is a peony")


