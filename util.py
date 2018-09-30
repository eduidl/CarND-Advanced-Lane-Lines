import cv2
import pickle

def read_mtx_dist():
    with open('calibration.p', 'rb') as f:
        calibrate = pickle.load(f)

    mtx = calibrate['mtx']
    dist = calibrate['dist']
    return mtx, dist

def save_gray_img(file_, img):
    cv2.imwrite(file_, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))

def save_rgb_img(file_, img):
    cv2.imwrite(file_, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
