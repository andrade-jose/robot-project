import cv2
import tensorflow as tf
import numpy as np

class MonoCamera:
    def __init__(self, config):
        self.cap = cv2.VideoCapture(config['index'])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['height'])
        self.depth_model = tf.keras.models.load_model('models/monodepth.h5')
    
    def get_frames(self):
        ret, color_img = self.cap.read()
        if not ret:
            raise IOError("Falha na captura")
        depth_map = self.estimate_depth(color_img)
        return color_img, depth_map
    
    def estimate_depth(self, color_img):
        img_preprocessed = cv2.resize(color_img, (256, 256))
        img_preprocessed = img_preprocessed / 255.0
        depth_map = self.depth_model.predict(np.expand_dims(img_preprocessed, axis=0))
        return depth_map[0,:,:,0]
    
    def stop(self):
        self.cap.release()