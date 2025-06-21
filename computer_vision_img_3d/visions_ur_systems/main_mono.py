import numpy as np
import cv2
import tensorflow as tf
from mono.mono_camera import MonoCamera
from detection.object_detector import ObjectDetector
from control.coordinate_transformer import CoordinateTransformer
from control.ur_controller import URController
import json
import time

class VisionToMotionController:
    def __init__(self, config):
        self.camera = MonoCamera(config['camera'])
        self.detector = ObjectDetector(config['model_path'])
        self.transformer = CoordinateTransformer(config['calibration_file'])
        self.robot = URController(config['robot_ip'])
        
    def run_single_cycle(self) -> bool:
        color_img, depth_map = self.camera.get_frames()
        detection = self.detector.detect_object(color_img, depth_map)
        
        if detection:
            robot_coords = self.transformer.camera_to_robot(detection['position'])
            self.robot.move_to(robot_coords)
            return True
        return False
    
    def shutdown(self):
        self.camera.stop()
        self.robot.disconnect()

MONO_CONFIG = {
    'camera': {
        'index': 0,
        'width': 640,
        'height': 480
    },
    'model_path': 'models/object_detector.h5',
    'calibration_file': 'config/mono_calibration.json',
    'robot_ip': '192.168.1.10'
}

if __name__ == "__main__":
    system = VisionToMotionController(MONO_CONFIG)
    try:
        while True:
            success = system.run_single_cycle()
            print(f"Status: {'Objeto detectado' if success else 'Busca...'}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Parando sistema...")
    finally:
        system.shutdown()