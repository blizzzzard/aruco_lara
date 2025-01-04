import os 
import sys 

import argparse

from utils import aruco_marker


def main(args):

    task = args.task

    aruco = aruco_marker(args)

    if task == "generate":
        aruco.marker_generation()

    elif task == "detection":
        aruco.marker_detection()

    elif task == "calibration":
        aruco.images_calibration()
        aruco.cam_calibration()

    elif task == "estimation":
        aruco.pose_estimation()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '-t', type=str,)
    parser.add_argument('--aruco-size', '-s', type=int, default=512,)
    parser.add_argument('--calibration-path', '-c', type=str, default='./calib_data',)
    parser.add_argument('--num-markers', '-n', type=int, default=1,)
    args = parser.parse_args()

    main(args)