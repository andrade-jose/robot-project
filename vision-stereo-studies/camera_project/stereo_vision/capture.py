import cv2
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
import time
from datetime import datetime

class StereoCameras:
    def __init__(self, left_cam_id=0, right_cam_id=2):
        """Inicializa câmeras com tratamento de erro robusto"""
        self._setup_logging()
        self.logger.info("Inicializando câmeras estéreo...")
        
        self.cap_left = cv2.VideoCapture(left_cam_id)
        self.cap_right = cv2.VideoCapture(right_cam_id)
        
        if not self._verify_cameras():
            raise RuntimeError("Falha na inicialização das câmeras")
            
        self._configure_cameras()
        self.rectification_maps = None
        self.logger.info("Câmeras inicializadas com sucesso")

    def _setup_logging(self):
        """Configura sistema de logging"""
        self.logger = logging.getLogger('StereoCameras')
        self.logger.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Log para arquivo (rotativo)
        file_handler = RotatingFileHandler('stereo_capture.log', maxBytes=1e6, backupCount=3)
        file_handler.setFormatter(formatter)
        
        # Log para console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _verify_cameras(self):
        """Verifica conexão com as câmeras"""
        for _ in range(3):  # Tentativas
            if self.cap_left.isOpened() and self.cap_right.isOpened():
                return True
            time.sleep(0.5)
        return False

    def _configure_cameras(self):
        """Configura parâmetros comuns das câmeras"""
        for cap in [self.cap_left, self.cap_right]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            cap.set(cv2.CAP_PROP_FOCUS, 0)
            cap.set(cv2.CAP_PROP_FPS, 30)

    def set_rectification_maps(self, maps):
        """Define mapas de retificação"""
        self.rectification_maps = maps
        self.logger.info("Mapas de retificação configurados")

    def get_frames(self, rectify=True):
        """Captura frames sincronizados com tratamento de erro"""
        for _ in range(3):  # Tentativas
            ret_left, frame_left = self.cap_left.read()
            ret_right, frame_right = self.cap_right.read()
            
            if not (ret_left and ret_right):
                self.logger.warning("Falha na captura - tentando reconectar")
                self._reconnect()
                continue
                
            if rectify and self.rectification_maps:
                frame_left = cv2.remap(frame_left, *self.rectification_maps['left'], cv2.INTER_LINEAR)
                frame_right = cv2.remap(frame_right, *self.rectification_maps['right'], cv2.INTER_LINEAR)
            
            return True, (frame_left, frame_right)
            
        return False, (None, None)

    def _reconnect(self):
        """Tenta reconectar às câmeras"""
        self.release()
        time.sleep(1)
        self.__init__(self.cap_left.get(cv2.CAP_PROP_HW_DEVICE_ID), 
                     self.cap_right.get(cv2.CAP_PROP_HW_DEVICE_ID))

    def release(self):
        """Libera recursos das câmeras"""
        self.cap_left.release()
        self.cap_right.release()
        self.logger.info("Câmeras liberadas")

    def __del__(self):
        self.release()