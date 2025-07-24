import numpy as np
import pickle
import os

class CoordinateTransformer:
    """Transforma coordenadas da câmera para o sistema do robô"""

    def __init__(self, calibration_file=None, gripper_offset_z=0.02):
        self.calibration_file = calibration_file or os.path.join('robotics_project','data', 'stereo_dataset', 'calib.pkl')
        self.gripper_offset_z = gripper_offset_z
        self.T = None  # Matriz 4x4

    def set_transformation_from_points(self, camera_points, robot_points):
        camera_points = np.asarray(camera_points)
        robot_points = np.asarray(robot_points)
        assert camera_points.shape == robot_points.shape
        assert len(camera_points) >= 3, "Mínimo de 3 pares de pontos necessários"

        # Coordenadas homogêneas
        ones = np.ones((camera_points.shape[0], 1))
        camera_h = np.hstack([camera_points, ones])  # Nx4

        # Resolve T por mínimos quadrados
        T, _, _, _ = np.linalg.lstsq(camera_h, robot_points, rcond=None)
        self.T = np.vstack([T.T, [0, 0, 0, 1]])  # 4x4 homogênea

    def transform_point(self, camera_point):
        if self.T is None:
            raise ValueError("Transformação não definida ou carregada.")
        cam_p = np.array([*camera_point, 1.0])
        robot_p = self.T @ cam_p
        robot_p[2] -= self.gripper_offset_z
        return robot_p[:3]

    def save_transformation(self):
        if self.T is None:
            raise ValueError("Transformação não definida")
        os.makedirs(os.path.dirname(self.calibration_file), exist_ok=True)
        with open(self.calibration_file, 'wb') as f:
            pickle.dump(self.T, f)

    def load_transformation(self, path=None):
        if path is None:
            path = self.transformation_path
        if not os.path.exists(path):
            raise FileNotFoundError("Arquivo de transformação não encontrado")
        self.T = np.load(path)

