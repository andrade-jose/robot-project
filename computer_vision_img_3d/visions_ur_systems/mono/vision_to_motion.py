from mono.mono_camera import MonoCamera
from detection.object_detector import ObjectDetector
from control.coordinate_transformer import CoordinateTransformer
from control.ur_controller import URController
import time

class VisionToMotionController:
    def __init__(self, camera: MonoCamera, 
                 detector: ObjectDetector,
                 transformer: CoordinateTransformer,
                 robot: URController):
        self.camera = camera
        self.detector = detector
        self.transformer = transformer
        self.robot = robot
        
    def run_single_cycle(self) -> bool:
        """Executa um ciclo completo de visão e movimento"""
        color_img, depth_map = self.camera.get_frames()
        detection = self.detector.detect_object(color_img, depth_map)
        
        if detection:
            # Transforma coordenadas e move robô
            robot_coords = self.transformer.camera_to_robot(detection['position'])
            orientation = self.transformer.get_orientation(robot_coords[2])
            target_pose = list(robot_coords) + list(orientation)
            
            self.robot.move_to(target_pose)
            return True
        return False
    
    def shutdown(self):
        """Encerra todos os componentes"""
        self.camera.stop()
        self.robot.disconnect()