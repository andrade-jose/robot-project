from camera.depth_camera import DepthCamera
from detection.object_detector import ObjectDetector
from control.coordinate_transformer import CoordinateTransformer
from control.ur_controller import URController
from typing import Dict
import time

class VisionToMotionController:
    """
    Classe principal que integra visão computacional e controle do robô
    
    Atributos:
        camera (DepthCamera): Instância da câmera
        detector (ObjectDetector): Detector de objetos
        transformer (CoordinateTransformer): Transformador de coordenadas
        robot (URController): Controlador do robô UR
    """
    
    def __init__(self, 
                camera_config: Dict, 
                model_path: str, 
                calibration_file: str, 
                robot_ip: str):
        """
        Inicializa todos os componentes do sistema
        
        Args:
            camera_config (Dict): Configuração da câmera
            model_path (str): Caminho para modelo de detecção
            calibration_file (str): Caminho para arquivo de calibração
            robot_ip (str): Endereço IP do robô UR
        """
        self.camera = DepthCamera(camera_config)
        self.detector = ObjectDetector(model_path)
        self.transformer = CoordinateTransformer(calibration_file)
        self.robot = URController(robot_ip)
        
        # Variáveis de estado
        self.current_object = None
        self.robot_pose = None
    
    def update_vision(self):
        """Atualiza detecção de objetos com os frames mais recentes"""
        color_img, depth_map = self.camera.get_aligned_frames()
        self.current_object = self.detector.detect_object(color_img, depth_map)
        return color_img, depth_map
    
    def calculate_target_poses(self, object_pos: np.ndarray) -> Dict:
        """
        Calcula poses para operação de pick-and-place
        
        Args:
            object_pos (np.ndarray): Posição do objeto no espaço da câmera
            
        Returns:
            Dict: Dicionário com poses:
                - 'approach': Pose de aproximação
                - 'target': Pose para pegar objeto
                - 'retreat': Pose de retirada
        """
        # Converter para espaço do robô
        robot_pos = self.transformer.camera_to_robot(object_pos)
        
        # Obter orientação baseada na altura do objeto
        orientation = self.transformer.calculate_gripper_orientation(robot_pos[2])
        
        # Criar poses
        target_pose = np.concatenate([robot_pos, orientation])
        approach_pose = target_pose.copy()
        approach_pose[2] += 0.1  # 10cm acima do objeto
        retreat_pose = approach_pose.copy()
        
        return {
            'approach': approach_pose,
            'target': target_pose,
            'retreat': retreat_pose
        }
    
    def execute_pick_sequence(self, poses: Dict):
        """Executa sequência completa de pick-and-place"""
        script = self.robot.generate_pick_script(
            poses['approach'],
            poses['target'],
            poses['retreat']
        )
        self.robot.execute_script(script)
    
    def run_single_cycle(self) -> bool:
        """
        Executa um ciclo completo de detecção e movimento
        
        Returns:
            bool: True se objeto foi detectado e movimento iniciado
        """
        # Atualizar detecção
        self.update_vision()
        
        if self.current_object:
            # Calcular poses
            poses = self.calculate_target_poses(self.current_object['position'])
            
            # Executar sequência
            self.execute_pick_sequence(poses)
            return True
        return False
    
    def shutdown(self):
        """Desliga todos os componentes do sistema"""
        self.camera.stop()
        self.robot.disconnect()