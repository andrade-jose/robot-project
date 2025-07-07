# stereo_vision/valid_image_pair.py

import cv2
import os
import numpy as np

def is_blurry(image, threshold=100.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var < threshold  # True = borrada

def is_well_exposed(image, min_mean=50, max_mean=200):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    return min_mean < mean_intensity < max_mean

def is_chessboard_far_from_border(corners, img_shape, margin=10):
    x_min = np.min(corners[:, 0, 0])
    x_max = np.max(corners[:, 0, 0])
    y_min = np.min(corners[:, 0, 1])
    y_max = np.max(corners[:, 0, 1])
    h, w = img_shape[:2]
    return (x_min > margin and x_max < (w - margin) and
            y_min > margin and y_max < (h - margin))

def is_valid_image_pair(img_left, img_right, chessboard_size):
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    retL, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size)
    retR, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size)

    if not (retL and retR):
        print("[INFO] Falha na detecção do tabuleiro em uma das imagens")
        return False

    if not (is_chessboard_far_from_border(corners_left, gray_left.shape) and
            is_chessboard_far_from_border(corners_right, gray_right.shape)):
        print("[INFO] Tabuleiro muito próximo da borda")
        return False

    if is_blurry(img_left) or is_blurry(img_right):
        print("[INFO] Imagem desfocada detectada")
        return False

    if not is_well_exposed(img_left) or not is_well_exposed(img_right):
        print("[INFO] Imagem com má exposição detectada")
        return False

    return True
