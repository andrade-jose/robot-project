import tensorflow as tf
import numpy as np
import cv2

class ObjectDetector:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.input_size = self.model.input_shape[1:3]
    
    def detect_object(self, color_img, depth_map):
        resized = cv2.resize(color_img, self.input_size)
        normalized = resized / 255.0
        pred = self.model.predict(np.expand_dims(normalized, axis=0))[0]
        
        if pred[4] < 0.5:
            return None
        
        x1, y1, x2, y2 = pred[:4]
        u = int((x1 + x2)/2 * color_img.shape[1] / self.input_size[1])
        v = int((y1 + y2)/2 * color_img.shape[0] / self.input_size[0])
        
        z = depth_map[v, u] if depth_map is not None else 0.5
        return {
            'position': (u, v, z),
            'confidence': pred[4]
        }