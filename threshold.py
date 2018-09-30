import cv2
import numpy as np

import util

class SobelThreshold:
    def __init__(self, kernel, thresh):
        self.kernel = kernel
        self.thresh = thresh

    def binarize(self, rgb, save=False):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (3,3), 0)

        sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=self.kernel)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        binary = np.zeros_like(scaled_sobel)
        binary[(scaled_sobel >= self.thresh[0]) & (scaled_sobel <= self.thresh[1])] = 255 

        if save: util.save_gray_img('writeup_images/sobel_binary.jpg', binary)

        return binary

class ColorThreshold:
    def __init__(self, white_thresh, yellow_thresh):
        self.white_thresh = white_thresh
        self.yellow_thresh = yellow_thresh

    def binarize(self, rgb, save=False):
        hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS)
        white  = cv2.inRange(hls, self.white_thresh[0], self.white_thresh[1])
        yellow = cv2.inRange(hls, self.yellow_thresh[0], self.yellow_thresh[1])
    
        binary = np.zeros_like(white)
        binary[(white > 0) | (yellow > 0)] = 255

        if save: util.save_gray_img('writeup_images/color_binary.jpg', binary)

        return binary

class CombinedThreshold:
    def __init__(self, sobel_kernel=3, sobel_thresh=(30, 100),
                 white_thresh=((0,210,0),(255,255,255)),
                 yellow_thresh=((15,30,115),(35,205,255))):
        self.sobel_thresh = SobelThreshold(kernel=sobel_kernel,
                                           thresh=sobel_thresh)
        self.color_thresh = ColorThreshold(white_thresh=white_thresh,
                                           yellow_thresh=yellow_thresh)

    def binarize(self, rgb, save=False):
        sobel_binary = self.sobel_thresh.binarize(rgb, save=save)
        color_binary = self.color_thresh.binarize(rgb, save=save)
        binary = np.zeros_like(color_binary)
        binary[(color_binary > 0) | (sobel_binary > 0)] = 255

        if save: util.save_gray_img('writeup_images/combined_binary.jpg', binary)

        return binary
