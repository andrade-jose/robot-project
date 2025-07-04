# stereo_vision/processor.py

import cv2
import os
import numpy as np
import pickle

from stereo_vision.disparity import DisparityCalculator
from stereo_vision.reconstruction import DepthReconstructor

class StereoProcessor:
    def __init__(self, calib_file):
        """
        Inicializa o processador carregando os parâmetros de calibração.

        Args:
            calib_file (str): caminho para o arquivo .pkl gerado na calibração.
        """
        with open(calib_file, 'rb') as f:
            data = pickle.load(f)

        self.left_map1 = data['left_map1']
        self.left_map2 = data['left_map2']
        self.right_map1 = data['right_map1']
        self.right_map2 = data['right_map2']
        self.Q = data['Q']

        self.disparity_calc = DisparityCalculator()
        self.reconstructor = DepthReconstructor(self.Q)

    def process_pair(self, img_left_path, img_right_path):
        """
        Processa um par estéreo completo.

        Args:
            img_left_path (str): caminho da imagem esquerda.
            img_right_path (str): caminho da imagem direita.

        Retorna:
            dict: com 'disparity', 'depth_map', 'points_3D'.
        """
        img_left = cv2.imread(img_left_path)
        img_right = cv2.imread(img_right_path)

        # Retificação
        img_left_rect = cv2.remap(img_left, self.left_map1, self.left_map2, cv2.INTER_LINEAR)
        img_right_rect = cv2.remap(img_right, self.right_map1, self.right_map2, cv2.INTER_LINEAR)

        # Converter para cinza
        gray_left = cv2.cvtColor(img_left_rect, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right_rect, cv2.COLOR_BGR2GRAY)

        # Calcular disparidade e profundidade
        disparity = self.disparity_calc.compute(gray_left, gray_right)
        points_3D = self.reconstructor.compute_point_cloud(disparity)
        depth_map = self.reconstructor.get_depth_map(points_3D)

        return {
            'disparity': disparity,
            'depth_map': depth_map,
            'points_3D': points_3D
        }

    def save_results(self, output_dir, base_name, disparity, depth_map):
        """
        Salva as imagens de disparidade e profundidade como PNG.

        Args:
            output_dir (str): diretório base.
            base_name (str): nome base do arquivo (sem extensão).
        """
        os.makedirs(output_dir, exist_ok=True)

        disp_norm = self.disparity_calc.normalize(disparity)
        depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        cv2.imwrite(os.path.join(output_dir, f'{base_name}_disparity.png'), disp_norm)
        cv2.imwrite(os.path.join(output_dir, f'{base_name}_depth.png'), depth_vis)
