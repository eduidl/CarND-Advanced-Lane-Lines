import cv2
import numpy as np
from pathlib import Path
import pickle
import re

def extract_num(path):
    pattern = r"calibration(\d+).jpg"
    return int(re.match(pattern, path.name).group(1))

def calibrate(nx=9, ny=6):
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    paths = sorted(Path('camera_cal').glob('calibration*.jpg'), key=extract_num)
    
    # Step through the list and search for chessboard corners
    done = False
    for i, path in enumerate(paths):
        img = cv2.imread(path.as_posix())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

        if not ret: continue

        cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
        if not done:
            cv2.imwrite('writeup_images/chessboard.jpg', img)
            done = True

        objpoints.append(objp)
        imgpoints.append(corners)
                
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                       gray.shape[::-1], None, None)

    img = cv2.imread('camera_cal/calibration1.jpg')
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite('writeup_images/distorted_chessboard.jpg', img)
    cv2.imwrite('writeup_images/undistorted_chessboard.jpg', undistorted)


    return mtx, dist

if __name__ == '__main__':
    mtx, dist = calibrate()
    dist_pickle = { 'mtx': mtx, 'dist': dist }

    with open('calibration.p', 'wb') as f:
        pickle.dump(dist_pickle, f)
