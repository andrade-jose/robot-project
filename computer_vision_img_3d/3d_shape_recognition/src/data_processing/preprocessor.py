import cv2
import numpy as np
import caer

class ImagePreprocessor:
    def __init__(self, img_size=(80, 80), channels=1):
        self.img_size = img_size
        self.channels = channels
    
    def prepare(self, img):
        """Pré-processa uma imagem para entrada no modelo"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, self.img_size)
        equalized = cv2.equalizeHist(resized)
        reshaped = caer.reshape(equalized, self.img_size, self.channels)
        reshaped = reshaped.astype('float32') / 255.0
        return reshaped.reshape(1, *self.img_size, self.channels), equalized