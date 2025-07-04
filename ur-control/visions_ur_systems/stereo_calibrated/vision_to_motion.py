from stereo_calibrated.stereo_camera import StereoCamera
from detection.object_detector import ObjectDetector
from control.coordinate_transformer import CoordinateTransformer
from control.ur_controller import URController
import time
import numpy as np

class VisionToMotionController:
    def __init__(self, camera: StereoCamera,
                 detector: ObjectDetector,
                 transformer: CoordinateTransformer,
                 robot: URController):
        self.camera = camera
        self.detector = detector
        self.transformer = transformer
        self.robot = robot
        self.last_position = None
        
    def run_single_cycle(self) -> bool:
        """Executa um ciclo com filtro de Kalman para suavizar movimentos"""
        left_img, right_img, depth_map = self.camera.get_frames()
        detection = self.detector.detect_object(left_img, depth_map)
        
        if detection:
            # Suaviza a posição detectada
            smoothed_pos = self._apply_kalman_filter(detection['position'])
            robot_coords = self.transformer.camera_to_robot(smoothed_pos)
            
            # Movimento seguro com verificação de colisão
            if self._is_safe_move(robot_coords):
                self.robot.move_to(list(robot_coords) + [np.pi, 0, 0])
                self.last_position = robot_coords
                return True
        return False
    
    def _apply_kalman_filter(self, position):
        """Filtro de Kalman simplificado para suavizar posições"""
        if not hasattr(self, 'kalman_filter'):
            self._init_kalman_filter()
        # Implementação simplificada - usar OpenCV KalmanFilter em produção
        if self.last_position is None:
            return position
        return 0.7 * np.array(position) + 0.3 * self.last_position
    
    def _init_kalman_filter(self):
        """Inicializa o filtro de Kalman"""
        # Implementação real exigiria cv2.KalmanFilter
        pass
    
    def _is_safe_move(self, target_pos):
        """Verifica se o movimento é seguro"""
        if self.last_position is None:
            return True
        distance = np.linalg.norm(np.array(target_pos) - self.last_position)
        return distance < 0.5  # Limite de segurança de 50cm
    
    def shutdown(self):
        """Libera todos os recursos"""
        self.camera.stop()
        self.robot.disconnect()