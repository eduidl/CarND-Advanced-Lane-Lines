import cv2
import numpy as np

import util
from threshold import CombinedThreshold

threshold = CombinedThreshold()

w = 1280
h = 720
dst_w = 500

LABELS = ['left', 'right']

class LaneDetector:
    def __init__(self, nwindows=9, margin_first=50, margin=10, minpix=50):
        self.nwindows = nwindows
        self.margin_first = margin_first
        self.margin = margin
        self.minpix = minpix
        self.set_transform_matrix()
        self.fit = {}
        self.is_first = True

    def set_transform_matrix(self):
        t_y = 445
        t_x_diff = 85
        b_x_diff = 1000

        src = np.float32([
            [w/2 - t_x_diff, t_y],
            [w/2 - b_x_diff, h],
            [w/2 + b_x_diff, h],
            [w/2 + t_x_diff, t_y]])
        dst = np.float32([[0, 0], [0, h], [dst_w, h], [dst_w, 0]])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, src)

    def warp(self, img):
        return cv2.warpPerspective(img, self.M, (dst_w, h), flags=cv2.INTER_LINEAR)

    def unwarp(self, img):
        return cv2.warpPerspective(img, self.M_inv, (w, h))

    def get_xy(self, nonzerox, nonzeroy, lane_inds):
        try:
            lane_inds = np.concatenate(lane_inds)
        except ValueError:
            pass

        return nonzerox[lane_inds], nonzeroy[lane_inds]

    # first time
    def find_lane_pixels(self, binary_warped):
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        midpoint = np.int(histogram.shape[0]//2)
        x_base = { 'left': np.argmax(histogram[:midpoint]),
                   'right': np.argmax(histogram[midpoint:]) + midpoint }

        window_height = np.int(binary_warped.shape[0]//self.nwindows)
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        x_current = { 'left': x_base['left'], 'right': x_base['right'] }

        lane_inds = { 'left': [], 'right': [] }

        for window in range(self.nwindows):
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height

            for label in LABELS:
                win_x_low = x_current[label] - self.margin_first
                win_x_high = x_current[label] + self.margin_first

                good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                             (nonzerox >= win_x_low) & (nonzerox < win_x_high)
                            ).nonzero()[0]

                lane_inds[label].append(good_inds)

                if len(good_inds) > self.minpix:
                    x_current[label] = np.int(np.mean(nonzerox[good_inds]))

        x, y = {}, {}
        for label in LABELS:
            x[label], y[label] = self.get_xy(nonzerox, nonzeroy, lane_inds[label])

        return x, y

    def get_xy2(self, nonzerox, nonzeroy, fit):
        lane_inds = ((nonzerox > (fit[0]*(nonzeroy**2) + fit[1]*nonzeroy + fit[2] - self.margin)) &
                     (nonzerox < (fit[0]*(nonzeroy**2) + fit[1]*nonzeroy + fit[2] + self.margin)))
        return nonzerox[lane_inds], nonzeroy[lane_inds]

    def find_lane_pixels_around_before(self, binary_warped):
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        x, y = {}, {}
        for label in LABELS:
            x[label], y[label] = self.get_xy2(nonzerox, nonzeroy, self.fit[label])

        return x, y
    
    def get_curvature(self, x, y, ploty):
        xm_per_pix = 3.7/250
        ym_per_pix = 30/720
        y_eval = np.max(ploty)
        
        scene_w = dst_w * xm_per_pix
        scene_h = h * ym_per_pix
        
        curverad, intercept = {}, {}
        for label in LABELS:
            fit_cr = np.polyfit(y[label]*ym_per_pix, x[label]*xm_per_pix, 2)
            curverad[label] = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
            intercept[label] = fit_cr[0]*scene_h**2 + fit_cr[1]*scene_h + fit_cr[2]
            
        center = (intercept['left'] + intercept['right'])/2.0
        car_pos = center - scene_w/2.0

        return curverad, car_pos

    def fit_polynomial(self, binary_warped, save=False):
        if self.is_first:
            x, y = self.find_lane_pixels(binary_warped)
        else:
            x, y = self.find_lane_pixels_around_before(binary_warped)

        for label in LABELS:
            self.fit[label] = np.polyfit(y[label], x[label], 2)

        fitx = {}
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        for label in LABELS:
            try:
                fitx[label] = self.fit[label][0]*ploty**2 + self.fit[label][1]*ploty + self.fit[label][2]
            except TypeError:
                print('The function failed to fit a line!')
                fitx[label] = 1*ploty**2 + 1*ploty

        if save:
            out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
            out_img[y['left'], x['left']] = [255, 0, 0]
            out_img[y['right'], x['right']] = [0, 0, 255]

            for label in LABELS:
                for (x_, y_) in zip(fitx[label], ploty):
                    cv2.circle(out_img, (int(x_), int(y_)), 1, (255, 255, 0), -1)
            util.save_rgb_img('writeup_images/lane_finding.jpg', out_img)
                
        curverad, car_pos = self.get_curvature(x, y, ploty)
            
        return fitx, ploty, curverad, car_pos
    
    def write_text(self, img, str, pos):
        cv2.putText(img, str, pos, cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
    
    def process(self, img, save=False):
        binary = threshold.binarize(img, save=save)
        binary_warped = self.warp(binary)
        if save:
            util.save_gray_img('writeup_images/binary_warped.jpg', binary_warped)
        fitx, ploty, curverad, car_pos = self.fit_polynomial(binary_warped, save=save)
        
        zeros = np.zeros_like(binary_warped)
        warped_lane = np.dstack((zeros, zeros, zeros))
    
        pts_left = np.array([np.transpose(np.vstack([fitx['left'], ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([fitx['right'], ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(warped_lane, np.int_([pts]), (0, 255, 0))
        lane_img = self.unwarp(warped_lane)
        
        output = cv2.addWeighted(img, 1.0, lane_img, 0.3, 0)
        
        self.write_text(output, "Left Lane Curvature: {:.2f}".format(curverad['left']), (30,30))
        self.write_text(output, "Right Lane Curvature: {:.2f}".format(curverad['right']), (30,60))
        self.write_text(output, 'Car Pos. from Center: {:.2f} m'.format(car_pos), (30,90))

        self.is_first = False
    
        return output
