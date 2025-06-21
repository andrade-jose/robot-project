# Adaptado para uso com duas câmeras simples (visão estéreo) + controle com RTDE
# Sistema completo com detecção, triangulação estéreo e controle do robô UR usando protocolo rtde_control

import cv2
import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, Dict
import json
import time
import pyrealsense2 as rs  # opcional se quiser suporte híbrido
import ur_rtde.rtde as rtde
import ur_rtde.rtde_config as rtde_config

class StereoCamera:
    def __init__(self, left_id=0, right_id=1):
        self.left_cam = cv2.VideoCapture(left_id)
        self.right_cam = cv2.VideoCapture(right_id)

        # Calibração exemplo
        self.focal_length = 700
        self.baseline = 0.06
        self.cx = 320
        self.cy = 240
        self.fx = self.focal_length
        self.fy = self.focal_length

        self.stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=15)

    def get_frames(self):
        _, left = self.left_cam.read()
        _, right = self.right_cam.read()
        return left, right

    def compute_depth_map(self, left, right):
        gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        disp = self.stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0
        return disp

    def disparity_to_depth(self, disparity):
        if disparity <= 0:
            return 0.0
        return (self.focal_length * self.baseline) / disparity

    def pixel_to_3d(self, u, v, disparity):
        z = self.disparity_to_depth(disparity)
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return (x, y, z)

class ObjectDetector:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.input_size = self.model.input_shape[1:3]

    def detect(self, image):
        resized = cv2.resize(image, self.input_size)
        normalized = resized / 255.0
        input_tensor = np.expand_dims(normalized, axis=0)
        pred = self.model.predict(input_tensor)[0]

        if pred[4] < 0.5:
            return None

        x1, y1, x2, y2 = pred[:4]
        u = int((x1 + x2)/2 * image.shape[1] / self.input_size[1])
        v = int((y1 + y2)/2 * image.shape[0] / self.input_size[0])
        return u, v

class CoordinateTransformer:
    def __init__(self, calibration_file):
        with open(calibration_file, 'r') as f:
            data = json.load(f)
        self.matrix = np.array(data['transformation_matrix'])
        self.offset = np.array(data.get('tool_offset', [0, 0, 0]))

    def camera_to_robot(self, pos):
        homog = np.append(pos, 1)
        robot_coords = self.matrix @ homog
        return robot_coords[:3] + self.offset

    def get_orientation(self, height):
        return (np.pi, 0, 0) if height > 0.3 else (0, np.pi/2, 0)

class URController:
    def __init__(self, robot_ip, config_file='rtde_config.xml'):
        self.rtde_c = rtde.RTDE(robot_ip, 30004)
        self.config = rtde_config.ConfigFile(config_file)
        self.state_names, self.state_types = self.config.get_recipe('state')
        self.setp_names, self.setp_types = self.config.get_recipe('setp')
        self.rtde_c.connect()
        self.rtde_c.send_output_setup(self.state_names, self.state_types)
        self.rtde_c.send_input_setup(self.setp_names, self.setp_types)

    def send_pose(self, pose):
        self.rtde_c.send(self.setp_names, self.setp_types, pose)

    def disconnect(self):
        self.rtde_c.disconnect()

class VisionToMotion:
    def __init__(self, model_path, calib_file, robot_ip):
        self.cam = StereoCamera()
        self.detector = ObjectDetector(model_path)
        self.transformer = CoordinateTransformer(calib_file)
        self.robot = URController(robot_ip)

    def run_cycle(self):
        left, right = self.cam.get_frames()
        disp = self.cam.compute_depth_map(left, right)
        obj = self.detector.detect(left)

        if obj:
            u, v = obj
            d = disp[v, u]
            pos_3d = self.cam.pixel_to_3d(u, v, d)
            pos_robot = self.transformer.camera_to_robot(pos_3d)
            orient = self.transformer.get_orientation(pos_robot[2])

            target = np.concatenate([pos_robot, orient])
            approach = target.copy()
            approach[2] += 0.1
            retreat = approach.copy()

            # Executa sequência via RTDE
            self.robot.send_pose(approach.tolist())
            time.sleep(2)
            self.robot.send_pose(target.tolist())
            time.sleep(2)
            self.robot.send_pose(retreat.tolist())
            return True
        return False

    def shutdown(self):
        self.cam.left_cam.release()
        self.cam.right_cam.release()
        self.robot.disconnect()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    system = VisionToMotion(
        model_path='object_detector.h5',
        calib_file='calibration.json',
        robot_ip='192.168.1.10'
    )

    try:
        while True:
            ok = system.run_cycle()
            print("Ciclo executado" if ok else "Sem detecção")
            time.sleep(1)
    except KeyboardInterrupt:
        print("Finalizando...")
    finally:
        system.shutdown()
