# stereo_vision/capture.py

import cv2

class StereoCameras:
    def __init__(self, left_cam_id=0, right_cam_id=1):
        """
        Inicializa as câmeras estéreo.

        Args:
            left_cam_id (int): ID da câmera esquerda.
            right_cam_id (int): ID da câmera direita.
        """
        self.cap_left = cv2.VideoCapture(left_cam_id)
        self.cap_right = cv2.VideoCapture(right_cam_id)

        if not self.cap_left.isOpened() or not self.cap_right.isOpened():
            raise RuntimeError("Não foi possível abrir as câmeras")

        # Mapas de retificação (opcionais)
        self.left_map1 = None
        self.left_map2 = None
        self.right_map1 = None
        self.right_map2 = None

    def set_rectification_maps(self, left_map1, left_map2, right_map1, right_map2):
        """
        Define os mapas para retificação das imagens.

        Args:
            left_map1, left_map2, right_map1, right_map2: mapas gerados após calibração.
        """
        self.left_map1 = left_map1
        self.left_map2 = left_map2
        self.right_map1 = right_map1
        self.right_map2 = right_map2

    def get_frames(self, rectify=False):
        """
        Captura um par de frames das câmeras.

        Args:
            rectify (bool): se True, aplica retificação.

        Returns:
            tuple: (frame_esquerda, frame_direita)
        """
        ret_left, frame_left = self.cap_left.read()
        ret_right, frame_right = self.cap_right.read()

        if not ret_left or not ret_right:
            raise RuntimeError("Falha ao capturar frames das câmeras")

        if rectify and self.left_map1 is not None:
            frame_left = cv2.remap(frame_left, self.left_map1, self.left_map2, cv2.INTER_LINEAR)
            frame_right = cv2.remap(frame_right, self.right_map1, self.right_map2, cv2.INTER_LINEAR)

        return frame_left, frame_right

    def release(self):
        """Libera os dispositivos de captura."""
        self.cap_left.release()
        self.cap_right.release()
