import cv2
import numpy as np
import json

class StereoCamera:
    def __init__(self, config):
        self.cam_left = cv2.VideoCapture(config['left_index'])
        self.cam_right = cv2.VideoCapture(config['right_index'])
        
        for cam in [self.cam_left, self.cam_right]:
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, config['width'])
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, config['height'])
        
        self.load_calibration(config['calibration_file'])
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16*5,
            blockSize=3
        )
    
    def load_calibration(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        self.Q = np.array(data['disparity_to_depth_matrix'])
    
    def get_frames(self):
        ret_l, left_img = self.cam_left.read()
        ret_r, right_img = self.cam_right.read()
        
        if not (ret_l and ret_r):
            raise IOError("Falha na captura estereoscópica")
        
        gray_l = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        
        disparity = self.stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0
        depth_map = cv2.reprojectImageTo3D(disparity, self.Q)[:,:,2]
        
        return left_img, right_img, depth_map
    
    def stop(self):
        self.cam_left.release()
        self.cam_right.release()