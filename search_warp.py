import argparse
import cv2
import numpy as np

from threshold import CombinedThreshold

def nothing(x):
    pass

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
img = cv2.imread(parser.parse_args().input)
img = CombinedThreshold().binarize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

w = 1280
w_half = w//2
h = 720

cv2.namedWindow('setting', cv2.WINDOW_NORMAL)
cv2.createTrackbar('t_y', 'setting', 445, h, nothing)
cv2.createTrackbar('t_x_diff', 'setting', 85, w_half, nothing)
cv2.createTrackbar('b_x_diff', 'setting', 1000, w_half*2, nothing)
cv2.createTrackbar('dst_w', 'setting',  500, w, nothing)

while(True):
    k = cv2.waitKey(1)
    if k == 27: break

    t_y = cv2.getTrackbarPos('t_y', 'setting')
    t_x_diff = cv2.getTrackbarPos('t_x_diff', 'setting')
    b_x_diff = cv2.getTrackbarPos('b_x_diff', 'setting')
    dst_w = cv2.getTrackbarPos('dst_w', 'setting')

    src = np.float32([
        [w_half - t_x_diff, t_y],
        [w_half - b_x_diff, h],
        [w_half + b_x_diff, h],
        [w_half + t_x_diff, t_y]
    ])
    dst = np.float32([
        [0, 0], [0, h], [dst_w, h], [dst_w, 0]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(img, M, (dst_w, h), flags=cv2.INTER_LINEAR)
    cv2.imshow('setting', cv2.cvtColor(cv2.hconcat([img, warp]), cv2.COLOR_GRAY2BGR))
