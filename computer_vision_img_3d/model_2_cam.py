"""
vision_ur_controller_standard.py
Sistema para robô UR com câmeras convencionais
"""

import numpy as np
import cv2
import ur_rtde.rtde as rtde
import ur_rtde.rtde_config as rtde_config
import tensorflow as tf
import time
import json
from typing import Tuple, Dict, Optional, List

class StereoCamera:
    """
    Classe para par de câmeras estereoscópicas
    """
    
    def __init__(self, config: Dict):
        self.cam_left = cv2.VideoCapture(config['left_index'])
        self.cam_right = cv2.VideoCapture(config['right_index'])
        
        # Configurações comuns
        for cam in [self.cam_left, self.cam_right]:
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, config['width'])
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, config['height'])
        
        # Parâmetros de calibração estereoscópica
        self.load_calibration(config['calibration_file'])
    
    def load_calibration(self, file_path: str):
        """Carrega parâmetros de calibração prévia"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        self.Q = np.array(data['disparity_to_depth_matrix'])
        self.stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    
    def get_frames(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ret_l, left_img = self.cam_left.read()
        ret_r, right_img = self.cam_right.read()
        
        if not (ret_l and ret_r):
            raise IOError("Falha na captura estereoscópica")
        
        # Calcula disparidade e profundidade
        gray_l = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        
        disparity = self.stereo.compute(gray_l, gray_r)
        depth_map = cv2.reprojectImageTo3D(disparity, self.Q)[:,:,2]
        
        return left_img, right_img, depth_map
    
    def stop(self):
        self.cam_left.release()
        self.cam_right.release()

class ObjectDetector:
    def detect_object(self, 
                    color_img: np.ndarray, 
                    depth_map: np.ndarray) -> Optional[Dict]:
        # Versão adaptada para funcionar com mapas de profundidade estimados
        # ... (código similar ao original)
        
        # Ajuste para profundidade estimada
        if depth_map.max() > 1.0:  # Normaliza se necessário
            depth_map = depth_map / depth_map.max()
        
        # Restante do código permanece igual

class CoordinateTransformer:
    def camera_to_robot(self, camera_coords: Tuple[float, float, float]) -> np.ndarray:
        """
        Versão adaptada para lidar com incertezas na profundidade:
        - Adiciona filtro de Kalman para suavização
        - Ajusta offset baseado na confiança da estimativa
        """
        # Filtro de Kalman (simplificado)
        if not hasattr(self, 'kalman_filter'):
            self.init_kalman()
        
        self.kalman_filter.predict()
        smoothed_coords = self.kalman_filter.correct(np.array(camera_coords))
        
        # Transformação homogênea
        homog_coords = np.append(smoothed_coords, 1)
        robot_coords = np.dot(self.transformation_matrix, homog_coords)
        
        return robot_coords[:3]
    
STEREO_CONFIG = {
    'cameras': {
        'left_index': 0,
        'right_index': 1,
        'width': 640,
        'height': 480
    },
    'calibration_file': 'stereo_calibration.json',
    # ... outros parâmetros
}

# Inicialização
system = VisionToMotionController(
    camera=StereoCamera(STEREO_CONFIG['cameras']),
    # ... outros componentes
)


def calibrate_stereo(images_left: List, images_right: List):
    """Calibração estereoscópica offline"""
    # Implementação completa requer:
    # 1. Detecção de cantos em pares de imagens
    # 2. Cálculo de parâmetros intrínsecos/extrínsecos
    # 3. Geração da matriz Q
    pass

def main():
    # Escolher configuração
    config = MONO_CONFIG  # Ou STEREO_CONFIG
    
    # Inicializar sistema
    system = VisionToMotionController(config)
    
    try:
        while True:
            success = system.run_single_cycle()
            print(f"Status: {'Objeto detectado' if success else 'Busca...'}")
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("Parando sistema...")
    finally:
        system.shutdown()

if __name__ == "__main__":
    main()

