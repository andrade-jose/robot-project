import cv2
import numpy as np
import glob
import json

def calibrate_stereo(images_left: list, images_right: list, 
                    chessboard_size: tuple, square_size: float):
    """
    Calibração estéreo completa usando pares de imagens de um tabuleiro de xadrez
    
    Args:
        images_left: Lista de caminhos para imagens da câmera esquerda
        images_right: Lista de caminhos para imagens da câmera direita
        chessboard_size: Tupla (linhas, colunas) do tabuleiro
        square_size: Tamanho do quadrado em metros
    """
    # Critério para encontrar cantos
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepara pontos do objeto 3D
    objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2) * square_size
    
    # Listas para armazenar pontos
    objpoints = []  # Pontos 3D no mundo real
    imgpoints_left = []  # Pontos 2D na imagem esquerda
    imgpoints_right = []  # Pontos 2D na imagem direita
    
    for img_left, img_right in zip(images_left, images_right):
        # Processa imagem esquerda
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, None)
        
        # Processa imagem direita
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size, None)
        
        if ret_left and ret_right:
            objpoints.append(objp)
            
            # Refina a localização dos cantos
            corners2_left = cv2.cornerSubPix(gray_left, corners_left, (11,11), (-1,-1), criteria)
            imgpoints_left.append(corners2_left)
            
            corners2_right = cv2.cornerSubPix(gray_right, corners_right, (11,11), (-1,-1), criteria)
            imgpoints_right.append(corners2_right)
    
    # Calibração individual das câmeras
    ret_left, mtx_left, dist_left, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints_left, gray_left.shape[::-1], None, None)
    
    ret_right, mtx_right, dist_right, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints_right, gray_right.shape[::-1], None, None)
    
    # Calibração estéreo
    flags = cv2.CALIB_FIX_INTRINSIC  # Usa parâmetros intrínsecos já calculados
    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtx_left, dist_left, mtx_right, dist_right,
        gray_left.shape[::-1], flags=flags)
    
    # Retificação estéreo
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        mtx_left, dist_left, mtx_right, dist_right,
        gray_left.shape[::-1], R, T)
    
    # Salva calibração em arquivo JSON
    calibration_data = {
        'left_camera_matrix': mtx_left.tolist(),
        'left_distortion': dist_left.tolist(),
        'right_camera_matrix': mtx_right.tolist(),
        'right_distortion': dist_right.tolist(),
        'rotation': R.tolist(),
        'translation': T.tolist(),
        'disparity_to_depth_matrix': Q.tolist()
    }
    
    with open('config/stereo_calibration.json', 'w') as f:
        json.dump(calibration_data, f)
    
    return calibration_data