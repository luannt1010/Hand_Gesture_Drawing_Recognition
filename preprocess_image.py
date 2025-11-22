import os
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def _preprocess(img_path, is_predict=False):
    try:
        img = cv2.imread(img_path)
    except Exception as e:
        print(e)
        return 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_inv = cv2.bitwise_not(gray)
    dist = cv2.distanceTransform((gray_inv > 0).astype(np.uint8), cv2.DIST_L2, 5)
    mean_thickness = np.mean(dist[dist > 0])
    if mean_thickness < 2:  
        kernel = np.ones((7, 7), np.uint8)
        gray_inv = cv2.dilate(gray_inv, kernel, iterations=1)
    coords = np.where(gray_inv > 20)
    if len(coords[0]) == 0:
        print("Not found drawing in the image")
        return None, None
    y_min, y_max = np.min(coords[0]), np.max(coords[0])
    x_min, x_max = np.min(coords[1]), np.max(coords[1])
    cropped = gray_inv[y_min:y_max + 1, x_min:x_max + 1]
    h, w = cropped.shape
    diff = abs(h - w)
    pad1, pad2 = diff // 2, diff - diff // 2
    if h > w:
        squared = np.pad(cropped, ((0, 0), (pad1, pad2)), constant_values=0)
    else:
        squared = np.pad(cropped, ((pad1, pad2), (0, 0)), constant_values=0)
    margin = int(squared.shape[0] * 0.15)
    padded = np.pad(squared, ((margin, margin), (margin, margin)), constant_values=0)
    resized = cv2.resize(padded, (64, 64), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    tensor = None
    if is_predict:
        # rgb = rgb.astype("float32") / 255.0
        tensor = np.expand_dims(rgb, 0)
    return rgb, tensor
    
def preprocess(img_path, save_path=None, name_file=None, is_predict=False):
    img, tensor = _preprocess(img_path, is_predict)
    if img is None:
        return None, None
    if not is_predict:
        if not save_path or not name_file:
            print("Invalid arguments")
        elif save_path and name_file:
            save_path = os.path.join(save_path, name_file)
            cv2.imwrite(save_path, img)
    return img, tensor