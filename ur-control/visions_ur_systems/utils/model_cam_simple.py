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

class MonoCamera:
    """
    Classe para câmera única com estimativa de profundidade por redes neurais
    """
    
    def __init__(self, config: Dict):
        self.cap = cv2.VideoCapture(config['index'])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['height'])
        
        # Carrega modelo de estimativa de profundidade (ex: MiDaS)
        self.depth_model = tf.keras.models.load_model('monodepth.h5')
    
    def get_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        ret, color_img = self.cap.read()
        if not ret:
            raise IOError("Falha na captura")
        
        # Estimativa de profundidade
        depth_map = self.estimate_depth(color_img)
        return color_img, depth_map
    
    def estimate_depth(self, color_img: np.ndarray) -> np.ndarray:
        """Usa rede neural para estimar mapa de profundidade"""
        img_preprocessed = preprocess_for_depth(color_img)  # Redimensiona/normaliza
        depth_map = self.depth_model.predict(img_preprocessed)
        return depth_map[0,:,:,0]  # Remove dimensões extras
    
    def stop(self):
        self.cap.release()

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
    
# Configuração mono
MONO_CONFIG = {
    'camera': {
        'index': 0,  # Webcam padrão
        'width': 640,
        'height': 480
    },
    'model_path': 'object_detector.h5',
    'calibration_file': 'mono_calibration.json',
    'robot_ip': '192.168.1.10'
}

# Inicialização
system = VisionToMotionController(
    camera=MonoCamera(MONO_CONFIG['camera']),
    detector=ObjectDetector(MONO_CONFIG['model_path']),
    transformer=CoordinateTransformer(MONO_CONFIG['calibration_file']),
    robot=URController(MONO_CONFIG['robot_ip'])
)

def preprocess_for_depth(img: np.ndarray) -> np.ndarray:
    """Prepara imagem para modelos de estimativa de profundidade"""
    img = cv2.resize(img, (256, 256))
    img = img / 255.0  # Normalização
    return np.expand_dims(img, axis=0)  # Adiciona dimensão do batch

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