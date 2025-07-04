import numpy as np
import json
from typing import Tuple

class CoordinateTransformer:
    """
    Classe para transformação entre sistemas de coordenadas
    
    Atributos:
        transformation_matrix (np.ndarray): Matriz 4x4 de transformação
    """
    
    def __init__(self, calibration_file: str):
        """
        Carrega matriz de transformação de arquivo JSON
        
        Args:
            calibration_file (str): Caminho para arquivo de calibração
        """
        with open(calibration_file, 'r') as f:
            calibration_data = json.load(f)
        
        self.transformation_matrix = np.array(calibration_data['transformation_matrix'])
        self.tool_offset = np.array(calibration_data.get('tool_offset', [0, 0, 0]))
    
    def camera_to_robot(self, camera_coords: Tuple[float, float, float]) -> np.ndarray:
        """
        Converte coordenadas da câmera para espaço do robô
        
        Args:
            camera_coords (Tuple[float, float, float]): (x, y, z) em metros
            
        Returns:
            np.ndarray: Coordenadas (x, y, z) no espaço do robô
        """
        homog_coords = np.append(camera_coords, 1)  # Coordenadas homogêneas
        robot_coords = np.dot(self.transformation_matrix, homog_coords)
        return robot_coords[:3] + self.tool_offset  # Aplica offset da ferramenta
    
    def calculate_gripper_orientation(self, 
                                   object_height: float, 
                                   approach_angle: float = 0) -> Tuple[float, float, float]:
        """
        Calcula orientação do gripper baseado na altura do objeto
        
        Args:
            object_height (float): Altura do objeto em metros
            approach_angle (float): Ângulo de aproximação em radianos
            
        Returns:
            Tuple[float, float, float]: Orientação (rx, ry, rz) em radianos
        """
        if object_height > 0.3:  # Objetos altos - abordagem vertical
            return (np.pi, 0, approach_angle)
        else:  # Objetos baixos - abordagem horizontal
            return (0, np.pi/2, approach_angle)
