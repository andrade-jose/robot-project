# stereo_vision/reconstruction.py

import cv2
import numpy as np

class DepthReconstructor:
    def __init__(self, Q_matrix):
        """
        Inicializa com a matriz Q da calibração estéreo.

        Args:
            Q_matrix (np.ndarray): matriz 4x4 para reprojeção 3D.
        """
        self.Q = Q_matrix

    def compute_point_cloud(self, disparity_map):
        """
        Converte o mapa de disparidade em pontos 3D.

        Args:
            disparity_map (np.ndarray): mapa de disparidade float32.

        Returns:
            np.ndarray: nuvem de pontos 3D (H x W x 3).
        """
        points_3D = cv2.reprojectImageTo3D(disparity_map, self.Q)
        return points_3D

    def get_depth_map(self, points_3D):
        """
        Extrai o mapa de profundidade (canal Z).

        Args:
            points_3D (np.ndarray): nuvem de pontos 3D.

        Returns:
            np.ndarray: mapa de profundidade (float).
        """
        return points_3D[:, :, 2]

    def filter_valid_points(self, points_3D, disparity_map):
        """
        Filtra pontos válidos (disparidade > 0).

        Args:
            points_3D (np.ndarray): nuvem de pontos.
            disparity_map (np.ndarray): mapa de disparidade.

        Returns:
            np.ndarray: pontos válidos Nx3.
        """
        mask = disparity_map > 0
        valid_points = points_3D[mask]
        return valid_points
