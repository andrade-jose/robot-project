import numpy as np
import pickle
import os

from utils.configuracao_projeto import configurar_caminho_projeto

class CoordinateTransformer:
    """Transforma coordenadas da câmera para o sistema do robô"""

    def __init__(self, calibration_file=None, gripper_offset_z=0.02, usar_ficticio=False):
        self.calibration_file = calibration_file or os.path.join('robotics_project','data', 'stereo_dataset', 'calib.pkl')
        self.gripper_offset_z = gripper_offset_z
        self.T = None

        if usar_ficticio:
            print("⚠️ Usando matriz identidade fictícia para transformação")
            self.T = np.eye(4)  # Força matriz identidade e não tenta carregar arquivo
        else:
            if os.path.exists(self.calibration_file):
                try:
                    self.load_transformation()
                except Exception as e:
                    print(f"⚠️ Erro ao carregar transformação: {e}")
                    print("🔄 Usando matriz identidade como fallback")
                    self.T = np.eye(4)
            else:
                print("❌ Arquivo calib.pkl não encontrado, usando matriz identidade")
                self.T = np.eye(4)

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
        """Transformar ponto da câmera para coordenadas do robô"""
        if self.T is None:
            print("⚠️ Matriz de transformação não carregada, usando identidade")
            self.T = np.eye(4)

        # Converter para array numpy se necessário
        if not isinstance(camera_point, np.ndarray):
            camera_point = np.array(camera_point)

        # Garantir que temos 3 coordenadas
        if len(camera_point) != 3:
            raise ValueError(f"Ponto deve ter 3 coordenadas, recebido: {len(camera_point)}")

        # Converter para coordenadas homogêneas
        cam_p = np.array([camera_point[0], camera_point[1], camera_point[2], 1.0])

        try:
            # Aplicar transformação
            robot_p = self.T @ cam_p
            # Retornar apenas as coordenadas x, y, z
            return robot_p[:3]
        except Exception as e:
            print(f"❌ Erro na transformação: {e}")
            print(f"   Forma da matriz T: {self.T.shape}")
            print(f"   Forma do ponto: {cam_p.shape}")
            # Retornar ponto original como fallback
            return camera_point

    def save_transformation(self):
        if self.T is None:
            raise ValueError("Transformação não definida")
        os.makedirs(os.path.dirname(self.calibration_file), exist_ok=True)
        with open(self.calibration_file, 'wb') as f:
            pickle.dump(self.T, f)
        print(f"✅ Transformação salva em: {self.calibration_file}")

    def load_transformation(self, path=None):
        """Versão simplificada do carregamento"""
        if path is None:
            path = self.calibration_file

        print(f"Carregando transformação de: {path}")

        if not os.path.exists(path):
            print("❌ Arquivo não encontrado, usando matriz identidade")
            self.T = np.eye(4)
            return

        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)

            # Casos principais apenas
            if isinstance(data, np.ndarray) and data.shape == (4, 4):
                self.T = data
                print("✅ Matriz 4x4 carregada")
            elif isinstance(data, dict) and 'T' in data:
                self.T = data['T']
                print("✅ Matriz carregada do dicionário")
            else:
                print("⚠️ Formato não reconhecido, usando identidade")
                self.T = np.eye(4)

        except Exception as e:
            print(f"⚠️ Erro ao carregar: {e}, usando matriz identidade")
            self.T = np.eye(4)
