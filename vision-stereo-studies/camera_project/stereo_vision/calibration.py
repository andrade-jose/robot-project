import cv2
import numpy as np
import glob
import pickle

class StereoCalibration:
    def __init__(self, chessboard_size=(8, 5), square_size=0.026):
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.objpoints = []
        self.imgpoints_left = []
        self.imgpoints_right = []
        self.mtx_left = None
        self.dist_left = None
        self.mtx_right = None
        self.dist_right = None
        self.R = None
        self.T = None
        self.E = None
        self.F = None
        self.P1 = None
        self.P2 = None
        self.Q = None
        self.left_map1 = None
        self.left_map2 = None
        self.right_map1 = None
        self.right_map2 = None
        self._first_left_image_path = None

    def _prepare_object_points(self):
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        return objp

    def load_images(self, left_images_dir, right_images_dir):
        images_left = sorted(glob.glob(left_images_dir + '/*.png')) + sorted(glob.glob(left_images_dir + '/*.jpg'))
        images_right = sorted(glob.glob(right_images_dir + '/*.png')) + sorted(glob.glob(right_images_dir + '/*.jpg'))
        if len(images_left) != len(images_right):
            raise RuntimeError("Número diferente de imagens esquerda e direita")
        self._first_left_image_path = images_left[0]
        return images_left, images_right

    def find_corners(self, images_left, images_right):
        objp = self._prepare_object_points()
        for img_left_path, img_right_path in zip(images_left, images_right):
            img_left = cv2.imread(img_left_path)
            img_right = cv2.imread(img_right_path)
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
            retL, corners_left = cv2.findChessboardCorners(gray_left, self.chessboard_size)
            retR, corners_right = cv2.findChessboardCorners(gray_right, self.chessboard_size)
            if retL and retR:
                criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
                corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
                corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
                self.objpoints.append(objp)
                self.imgpoints_left.append(corners_left)
                self.imgpoints_right.append(corners_right)

    def calibrate(self):
        img_shape = self.imgpoints_left[0].shape[1::-1]
        _, self.mtx_left, self.dist_left, _, _ = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_left, img_shape, None, None)
        _, self.mtx_right, self.dist_right, _, _ = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_right, img_shape, None, None)

    def stereo_calibrate(self):
        img_shape = self.imgpoints_left[0].shape[1::-1]
        criteria = (cv2.TermCriteria_MAX_ITER + cv2.TermCriteria_EPS, 30, 1e-6)
        _, self.mtx_left, self.dist_left, self.mtx_right, self.dist_right, \
        self.R, self.T, self.E, self.F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_left, self.imgpoints_right,
            self.mtx_left, self.dist_left,
            self.mtx_right, self.dist_right,
            img_shape, criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC)

    def rectify(self):
        img_sample = cv2.imread(self._first_left_image_path)
        img_shape = (img_sample.shape[1], img_sample.shape[0])
        R1, R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            self.mtx_left, self.dist_left,
            self.mtx_right, self.dist_right,
            img_shape, self.R, self.T, alpha=1.0)
        self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(
            self.mtx_left, self.dist_left, R1, self.P1, img_shape, cv2.CV_16SC2)
        self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(
            self.mtx_right, self.dist_right, R2, self.P2, img_shape, cv2.CV_16SC2)

    def save_calibration(self, path):
        data = {
            'mtx_left': self.mtx_left,
            'dist_left': self.dist_left,
            'mtx_right': self.mtx_right,
            'dist_right': self.dist_right,
            'R': self.R,
            'T': self.T,
            'E': self.E,
            'F': self.F,
            'P1': self.P1,
            'P2': self.P2,
            'Q': self.Q,
            'left_map1': self.left_map1,
            'left_map2': self.left_map2,
            'right_map1': self.right_map1,
            'right_map2': self.right_map2,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load_calibration(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.__dict__.update(data)
