from stereo_simple.stereo_camera import StereoCamera
from detection.object_detector import ObjectDetector
from control.coordinate_transformer import CoordinateTransformer
from control.ur_controller import URController
import time

class VisionToMotionController:
    def __init__(self, camera: StereoCamera,
                 detector: ObjectDetector,
                 transformer: CoordinateTransformer,
                 robot: URController):
        self.camera = camera
        self.detector = detector
        self.transformer = transformer
        self.robot = robot
        
    def run_single_cycle(self) -> bool:
        """Executa um ciclo de detecção e movimento"""
        left_img, right_img, depth_map = self.camera.get_frames()
        detection = self.detector.detect_object(left_img, depth_map)
        
        if detection:
            # Converte coordenadas e controla robô
            robot_coords = self.transformer.camera_to_robot(detection['position'])
            orientation = self.transformer.get_orientation(robot_coords[2])
            
            # Gera posições de abordagem e retirada
            approach = robot_coords.copy()
            approach[2] += 0.1  # 10cm acima
            
            self.robot.move_to(list(approach) + list(orientation))
            time.sleep(1)
            self.robot.move_to(list(robot_coords) + list(orientation))
            time.sleep(1)
            self.robot.move_to(list(approach) + list(orientation))
            
            return True
        return False
    
    def shutdown(self):
        """Libera recursos"""
        self.camera.stop()
        self.robot.disconnect()