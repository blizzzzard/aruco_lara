import os

import cv2 as cv
from cv2 import aruco
import numpy as np



class aruco_marker(object):
    def __init__(self, config) -> None:
            
        self.config = config

        self.marker_size = config.aruco_size
        self.num_markers = config.num_markers
        self.calibration_path = config.calibration_path
        
    def marker_generation(self,):
        # dictionary to specify type of the marker
        marker_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        os.makedirs("./markers")

        # MARKER_ID = 0
        MARKER_SIZE = self.marker_size  # pixels

        # generating unique IDs using for loop
        for id in range(self.num_markers):  # genereting 20 markers
            # using funtion to draw a marker
            marker_image = aruco.generateImageMarker(marker_dict, id, MARKER_SIZE)
            #cv.imshow("img", marker_image)
            cv.imwrite(f"markers/marker_{id}.png", marker_image)
            # cv.waitKey(0)
            # break

    def marker_detection(self,):
        marker_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        param_markers = aruco.DetectorParameters()

        # utilizes default camera/webcam driver
        cap = cv.VideoCapture(0)

        # iterate through multiple frames, in a live video feed
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # turning the frame to grayscale-only (for efficiency)
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            marker_corners, marker_IDs, reject = aruco.detectMarkers(
                gray_frame, marker_dict, parameters=param_markers
            )
            # getting conrners of markers
            if marker_corners:
                for ids, corners in zip(marker_IDs, marker_corners):
                    cv.polylines(
                        frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA
                    )
                    corners = corners.reshape(4, 2)
                    corners = corners.astype(int)
                    top_right = corners[0].ravel()
                    top_left = corners[1].ravel()
                    bottom_right = corners[2].ravel()
                    bottom_left = corners[3].ravel()
                    cv.putText(
                        frame,
                        f"id: {ids[0]}",
                        top_right,
                        cv.FONT_HERSHEY_PLAIN,
                        1.3,
                        (200, 100, 0),
                        2,
                        cv.LINE_AA,
                    )
                    # print(ids, "  ", corners)
            cv.imshow("frame", frame)
            key = cv.waitKey(1)
            if key == ord("q"):
                break
        cap.release()
        cv.destroyAllWindows()


    def images_calibration(self,):
        Chess_Board_Dimensions = (9, 6)

        n = 0  # image counter

        # checks images dir is exist or not
        image_path = "images"

        Dir_Check = os.path.isdir(image_path)

        if not Dir_Check:  # if directory does not exist, a new one is created
            os.makedirs(image_path)
            print(f'"{image_path}" Directory is created')
        else:
            print(f'"{image_path}" Directory already exists.')

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


        def detect_checker_board(image, grayImage, criteria, boardDimension):
            ret, corners = cv.findChessboardCorners(grayImage, boardDimension)
            if ret == True:
                corners1 = cv.cornerSubPix(grayImage, corners, (3, 3), (-1, -1), criteria)
                image = cv.drawChessboardCorners(image, boardDimension, corners1, ret)

            return image, ret


        cap = cv.VideoCapture(0)

        while True:
            _, frame = cap.read()
            copyFrame = frame.copy()
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            image, board_detected = detect_checker_board(
                frame, gray, criteria, Chess_Board_Dimensions
            )
            # print(ret)
            cv.putText(
                frame,
                f"saved_img : {n}",
                (30, 40),
                cv.FONT_HERSHEY_PLAIN,
                1.4,
                (0, 255, 0),
                2,
                cv.LINE_AA,
            )

            cv.imshow("frame", frame)
            # copyframe; without augmentation
            cv.imshow("copyFrame", copyFrame)

            key = cv.waitKey(1)

            if key == ord("q"):
                break
            if key == ord("s") and board_detected == True:
                # the checker board image gets stored
                cv.imwrite(f"{image_path}/image{n}.png", copyFrame)

                print(f"saved image number {n}")
                n += 1  # the image counter: incrementing
        cap.release()
        cv.destroyAllWindows()

        print("Total saved Images:", n)


    def cam_calibration(self,):
        # Chess/checker board size, dimensions
        CHESS_BOARD_DIM = (9, 6)

        # The size of squares in the checker board design.
        SQUARE_SIZE = 22  # millimeters (change it according to printed size)

        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


        calib_data_path = self.calibration_path
        CHECK_DIR = os.path.isdir(calib_data_path)


        # saving the image / camera calibration data

        if not CHECK_DIR:
            os.makedirs(calib_data_path)
            print(f'"{calib_data_path}" Directory is created')

        else:
            print(f'"{calib_data_path}" Directory already Exists.')

        # prepare object points, i.e. (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        obj_3D = np.zeros((CHESS_BOARD_DIM[0] * CHESS_BOARD_DIM[1], 3), np.float32)

        obj_3D[:, :2] = np.mgrid[0 : CHESS_BOARD_DIM[0], 0 : CHESS_BOARD_DIM[1]].T.reshape(
            -1, 2
        )
        obj_3D *= SQUARE_SIZE
        print(obj_3D)

        # Arrays to store object points and image points from all the given images.
        obj_points_3D = []  # 3d point in real world space
        img_points_2D = []  # 2d points in image plane

        # The images directory path
        image_dir_path = "images"

        files = os.listdir(image_dir_path)  # list of names of all the files present
        for file in files:
            print(file)
            imagePath = os.path.join(image_dir_path, file)
            # print(imagePath)

            image = cv.imread(imagePath)
            grayScale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(image, CHESS_BOARD_DIM, None)
            if ret == True:
                obj_points_3D.append(obj_3D)
                corners2 = cv.cornerSubPix(grayScale, corners, (3, 3), (-1, -1), criteria)
                img_points_2D.append(corners2)

                img = cv.drawChessboardCorners(image, CHESS_BOARD_DIM, corners2, ret)

        cv.destroyAllWindows()
        # h, w = image.shape[:2]
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
            obj_points_3D, img_points_2D, grayScale.shape[::-1], None, None
        )
        print("calibrated")

        print("dumping the data into one files using numpy ")
        np.savez(
            f"{calib_data_path}/MultiMatrix",
            camMatrix=mtx,
            distCoef=dist,
            rVector=rvecs,
            tVector=tvecs,
        )

        print("-------------------------------------------")

        print("loading data stored using numpy savez function\n \n \n")

    

    def pose_estimation(self,):

        # load in the calibration data
        calib_data_path = f"{self.calibration_path}/MultiMatrix.npz"

        calib_data = np.load(calib_data_path)
        print(calib_data.files)

        cam_mat = calib_data["camMatrix"]
        dist_coef = calib_data["distCoef"]
        r_vectors = calib_data["rVector"]
        t_vectors = calib_data["tVector"]

        MARKER_SIZE = 6  # centimeters (measure your printed marker size)

        marker_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

        param_markers = aruco.DetectorParameters()

        cap = cv.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            marker_corners, marker_IDs, reject = aruco.detectMarkers(
                gray_frame, marker_dict, parameters=param_markers
            )
            if marker_corners:
                rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
                    marker_corners, MARKER_SIZE, cam_mat, dist_coef
                )
                total_markers = range(0, marker_IDs.size)
                for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):
                    cv.polylines(
                        frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA
                    )
                    corners = corners.reshape(4, 2)
                    corners = corners.astype(int)
                    top_right = corners[0].ravel()
                    top_left = corners[1].ravel()
                    bottom_right = corners[2].ravel()
                    bottom_left = corners[3].ravel()

                    # Calculating the distance
                    distance = np.sqrt(
                        tVec[i][0][2] ** 2 + tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2
                    )
                    # Draw the pose of the marker
                    point = cv.drawFrameAxes(frame, cam_mat, dist_coef, rVec[i], tVec[i], 4, 4)
                    cv.putText(
                        frame,
                        f"id: {ids[0]} Dist: {round(distance, 2)}",
                        top_right,
                        cv.FONT_HERSHEY_PLAIN,
                        1.3,
                        (0, 0, 255),
                        2,
                        cv.LINE_AA,
                    )
                    cv.putText(
                        frame,
                        f"x:{round(tVec[i][0][0],1)} y: {round(tVec[i][0][1],1)} ",
                        bottom_right,
                        cv.FONT_HERSHEY_PLAIN,
                        1.0,
                        (0, 0, 255),
                        2,
                        cv.LINE_AA,
                    )
                    # print(ids, "  ", corners)
            cv.imshow("frame", frame)
            key = cv.waitKey(1)
            if key == ord("q"):
                break
        cap.release()
        cv.destroyAllWindows()