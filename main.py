import argparse
import cv2
from moviepy.editor import VideoFileClip
from pathlib import Path

import util
from lane_detector import LaneDetector

mtx, dist = util.read_mtx_dist()
detector = LaneDetector()

def process(img, save=False):
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    if save:
        util.save_rgb_img('writeup_images/undistorted.jpg', undistorted)
    return detector.process(img, save=save)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    # Save intermediate images. This mode is available wheen only single image process
    parser.add_argument('--save', action='store_true') 
    args = parser.parse_args()

    ext = Path(args.input).suffix

    if ext in ['.mp4']:
        clip = VideoFileClip(args.input)
        project_clip = clip.fl_image(process)
        project_clip.write_videofile(args.output, audio=False)
    elif ext in ['.jpg', '.png']:
        img = cv2.cvtColor(cv2.imread(args.input), cv2.COLOR_BGR2RGB)
        output = process(img, save=args.save)
        util.save_rgb_img(args.output, output)
    else:
        print("{} is not supported.".format(ext))
