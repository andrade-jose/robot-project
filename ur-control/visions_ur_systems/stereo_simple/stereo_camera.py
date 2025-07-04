import cv2
import numpy as np

class StereoCamera:
    def __init__(self, config):
        self.cam_left = cv2.VideoCapture(config['left_index'])
        self.cam_right = cv2.VideoCapture(config['right_index'])
        
        for cam in [self.cam_left, self.cam_right]:
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, config['width'])
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, config['height'])
        
        self.focal_length = 700
        self.baseline = 0.06
        self.stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=15)
    
    def get_frames(self):
        ret_l, left_img = self.cam_left.read()
        ret_r, right_img = self.cam_right.read()
        
        if not (ret_l and ret_r):
            raise IOError("Falha na captura estereoscópica")
        
        gray_l = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        disparity = self.stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0
        depth_map = (self.focal_length * self.baseline) / (disparity + 1e-6)
        
        return left_img, right_img, depth_map
    
    def stop(self):
        self.cam_left.release()
        self.cam_right.release()