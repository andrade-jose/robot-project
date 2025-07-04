# stereo_vision/disparity.py

import cv2
import numpy as np

class DisparityCalculator:
    def __init__(self, num_disparities=16*5, block_size=7):
        """
        Inicializa o objeto para calcular disparidade usando StereoSGBM.

        Args:
            num_disparities (int): faixa máxima de disparidade (múltiplo de 16).
            block_size (int): tamanho do bloco para correspondência (ímpar, 3-11).
        """
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * 3 * block_size ** 2,
            P2=32 * 3 * block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

    def compute(self, left_img_gray, right_img_gray):
        """
        Calcula o mapa de disparidade a partir das imagens cinza retificadas.

        Args:
            left_img_gray (np.ndarray): imagem esquerda em escala de cinza.
            right_img_gray (np.ndarray): imagem direita em escala de cinza.

        Returns:
            np.ndarray: mapa de disparidade float32.
        """
        disparity = self.matcher.compute(left_img_gray, right_img_gray).astype(np.float32) / 16.0
        return disparity

    def normalize(self, disparity):
        """
        Normaliza o mapa de disparidade para faixa 0-255 para visualização.

        Args:
            disparity (np.ndarray): mapa bruto de disparidade.

        Returns:
            np.ndarray: imagem normalizada uint8.
        """
        disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        disp_norm = np.uint8(disp_norm)
        return disp_norm
