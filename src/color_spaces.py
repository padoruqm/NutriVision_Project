import cv2
import numpy as np

def convert_color_spaces(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    return rgb, hsv, lab

def normalize_image(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    lab[:,:,0] /= 100.0
    lab[:,:,1] = (lab[:,:,1] + 128) / 255.0
    lab[:,:,2] = (lab[:,:,2] + 128) / 255.0
    return lab
