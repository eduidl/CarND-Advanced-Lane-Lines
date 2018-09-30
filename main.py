import argparse
import cv2
from moviepy.editor import VideoFileClip
import numpy as np
import pickle

from lane_detector import LaneDetector
from threshold import SobelThreshold
from threshold import ColorThreshold

sobel_threshold = SobelThreshold(thresh=(30,100))
color_threshold = ColorThreshold()
detector = LaneDetector(margin=150)

with open('calibrate.p', 'rb') as f:
    calibrate = pickle.load(f)

mtx = calibrate['mtx']
dist = calibrate['dist']

def process(img):
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    color_filtered = color_threshold.threshold(undistorted)
    sobel_filtered = sobel_threshold.threshold(undistorted)
    filter_ = np.zeros_like(color_filtered)
    filter_[(color_filtered > 0) | (sobel_filtered > 0)] = 1
    return detector.process(img, filter_)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    clip = VideoFileClip(args.input)
    project_clip = clip.fl_image(process)
    project_clip.write_videofile(args.output, audio=False)
