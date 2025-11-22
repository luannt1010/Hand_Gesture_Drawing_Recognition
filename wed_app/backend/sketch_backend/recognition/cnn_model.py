import tensorflow as tf
import os
import numpy as np

model_path = r"C:\Users\ntlua\Desktop\src\models\best_model3.h5"
model = tf.keras.models.load_model(model_path)
print("Load model successfully")