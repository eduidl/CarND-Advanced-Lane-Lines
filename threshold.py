import cv2
import numpy as np

class SobelThreshold:
    def __init__(self, sobel_kernel=3, thresh=(0, 255)):
        self.sobel_kernel = sobel_kernel
        self.thresh = thresh

    def abs_threshold(self, sobel):
        
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel > self.abs_thresh[0]) & 
                      (scaled_sobel < self.abs_thresh[1])] = 1

        return binary_output

    def threshold(self, rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (3,3), 0)

        sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        binary = np.zeros_like(scaled_sobel)
        binary[(scaled_sobel >= self.thresh[0]) & (scaled_sobel <= self.thresh[1])] = 1

        return binary

class ColorThreshold:
    def __init__(self, white_thresh=((0,210,0),(255,255,255)),
                 yellow_thresh=((15,30,115),(35,205,255))):
        self.white_thresh = white_thresh
        self.yellow_thresh = yellow_thresh

    def threshold(self, rgb):
        hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS)
        white  = cv2.inRange(hls, self.white_thresh[0], self.white_thresh[1])
        yellow = cv2.inRange(hls, self.yellow_thresh[0], self.yellow_thresh[1])
    
        binary = np.zeros_like(white)
        binary[(white > 0) | (yellow > 0)] = 1

        return binary
