import cv2
import numpy as np
import sys

def nothing(x):
    pass

def gray2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

file_ = sys.argv[1]
src = cv2.imread("{}".format(file_))
hls = cv2.cvtColor(src, cv2.COLOR_BGR2HLS)

cv2.namedWindow('setting', cv2.WINDOW_NORMAL)
cv2.createTrackbar('white H min', 'setting',   0, 255, nothing)
cv2.createTrackbar('white H max', 'setting', 255, 255, nothing)
cv2.createTrackbar('white L min', 'setting', 200, 255, nothing)
cv2.createTrackbar('white L max', 'setting', 255, 255, nothing)
cv2.createTrackbar('white S min', 'setting',   0, 255, nothing)
cv2.createTrackbar('white S max', 'setting', 255, 255, nothing)
cv2.createTrackbar('yellow H min', 'setting',  15, 255, nothing)
cv2.createTrackbar('yellow H max', 'setting',  35, 255, nothing)
cv2.createTrackbar('yellow L min', 'setting',  30, 255, nothing)
cv2.createTrackbar('yellow L max', 'setting', 205, 255, nothing)
cv2.createTrackbar('yellow S min', 'setting', 115, 255, nothing)
cv2.createTrackbar('yellow S max', 'setting', 255, 255, nothing)

while(True):
    k = cv2.waitKey(1)
    if k == 27:
        break;
    w_H_min = cv2.getTrackbarPos('white H min', 'setting')
    w_H_max  = cv2.getTrackbarPos('white H max', 'setting')
    w_L_min  = cv2.getTrackbarPos('white L min', 'setting')
    w_L_max  = cv2.getTrackbarPos('white L max', 'setting')
    w_S_min  = cv2.getTrackbarPos('white S min', 'setting')
    w_S_max  = cv2.getTrackbarPos('white S max', 'setting')
    y_H_min  = cv2.getTrackbarPos('yellow H min', 'setting')
    y_H_max  = cv2.getTrackbarPos('yellow H max', 'setting')
    y_L_min  = cv2.getTrackbarPos('yellow L min', 'setting')
    y_L_max  = cv2.getTrackbarPos('yellow L max', 'setting')
    y_S_min  = cv2.getTrackbarPos('yellow S min', 'setting')
    y_S_max  = cv2.getTrackbarPos('yellow S max', 'setting')

    white = cv2.inRange(hls, (w_H_min, w_L_min, w_S_min), (w_H_max, w_L_max, w_S_max))
    yellow = cv2.inRange(hls, (y_H_min, y_L_min, y_S_min), (y_H_max, y_L_max, y_S_max))
    result = cv2.bitwise_or(white, yellow)

    row1 = cv2.hconcat([src, gray2bgr(white)])
    row2 = cv2.hconcat([gray2bgr(yellow), gray2bgr(result)])

    cv2.imshow('setting', cv2.vconcat([row1, row2]))

cv2.destroyAllWindows()
