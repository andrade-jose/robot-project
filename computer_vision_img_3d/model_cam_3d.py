"""
vision_ur_controller.py
Sistema completo para controle de robô UR usando visão 3D da câmera Balmer
"""

import numpy as np
import cv2
import pyrealsense2 as rs
import ur_rtde.rtde as rtde
import ur_rtde.rtde_config as rtde_config
import tensorflow as tf
import time
import json
from typing import Tuple, Dict, Optional

class DepthCamera:
    """
    Classe para captura e processamento de dados da câmera Balmer/RealSense
    
    Atributos:
        pipeline (rs.pipeline): Pipeline de captura de frames
        depth_scale (float): Fator de escala para valores de profundidade
        intrinsics (rs.intrinsics): Parâmetros intrínsecos da câmera
    """
    
    def __init__(self, config: Dict):
        """
        Inicializa a câmera com configurações específicas
        
        Args:
            config (Dict): Dicionário com configurações:
                - width: Largura da imagem
                - height: Altura da imagem
                - fps: Taxa de quadros
        """
        self.pipeline = rs.pipeline()
        rs_config = rs.config()
        
        # Habilitar streams de profundidade e cor
        rs_config.enable_stream(rs.stream.depth, 
                               config['width'], 
                               config['height'], 
                               rs.format.z16, 
                               config['fps'])
        rs_config.enable_stream(rs.stream.color, 
                               config['width'], 
                               config['height'], 
                               rs.format.bgr8, 
                               config['fps'])
        
        # Iniciar pipeline
        profile = self.pipeline.start(rs_config)
        
        # Obter parâmetros da câmera
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    
    def get_aligned_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Captura frames alinhados de cor e profundidade
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (color_image, depth_image)
        """
        frames = self.pipeline.wait_for_frames()
        
        # Alinhar frames de profundidade ao frame de cor
        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)
        
        # Obter frames alinhados
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        # Converter para arrays numpy
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        return color_image, depth_image
    
    def pixel_to_3d(self, depth_map: np.ndarray, u: int, v: int) -> Tuple[float, float, float]:
        """
        Converte coordenadas de pixel para coordenadas 3D no espaço da câmera
        
        Args:
            depth_map (np.ndarray): Mapa de profundidade
            u (int): Coordenada x do pixel
            v (int): Coordenada y do pixel
            
        Returns:
            Tuple[float, float, float]: Coordenadas (x, y, z) em metros
        """
        depth = depth_map[v, u] * self.depth_scale
        x = (u - self.intrinsics.ppx) * depth / self.intrinsics.fx
        y = (v - self.intrinsics.ppy) * depth / self.intrinsics.fy
        return (x, y, depth)
    
    def stop(self):
        """Para a captura de frames"""
        self.pipeline.stop()


class ObjectDetector:
    """
    Classe para detecção de objetos usando redes neurais
    
    Atributos:
        model (tf.keras.Model): Modelo de detecção carregado
        input_size (Tuple[int, int]): Tamanho esperado da imagem de entrada
    """
    
    def __init__(self, model_path: str):
        """
        Carrega modelo de detecção pré-treinado
        
        Args:
            model_path (str): Caminho para o arquivo do modelo
        """
        self.model = tf.keras.models.load_model(model_path)
        self.input_size = self.model.input_shape[1:3]  # (height, width)
    
    def detect_object(self, 
                     color_img: np.ndarray, 
                     depth_map: np.ndarray) -> Optional[Dict]:
        """
        Detecta objeto principal na cena e retorna sua posição 3D
        
        Args:
            color_img (np.ndarray): Imagem colorida
            depth_map (np.ndarray): Mapa de profundidade correspondente
            
        Returns:
            Optional[Dict]: Dicionário com:
                - 'position': (x, y, z) em metros
                - 'bbox': [x1, y1, x2, y2] em pixels
                ou None se nenhum objeto for detectado
        """
        # Pré-processamento da imagem
        img_resized = cv2.resize(color_img, self.input_size)
        img_normalized = img_resized / 255.0
        input_tensor = np.expand_dims(img_normalized, axis=0)
        
        # Predição do modelo
        predictions = self.model.predict(input_tensor)
        
        # Parse das predições (assumindo modelo retorna [x1, y1, x2, y2, confiança])
        if predictions[0][4] < 0.5:  # Threshold de confiança
            return None
            
        bbox = predictions[0][:4]
        
        # Calcular centro do bounding box na imagem original
        scale_x = color_img.shape[1] / self.input_size[1]
        scale_y = color_img.shape[0] / self.input_size[0]
        
        x1, y1, x2, y2 = bbox
        u = int(((x1 + x2) / 2) * scale_x)
        v = int(((y1 + y2) / 2) * scale_y)
        
        # Obter coordenadas 3D
        x, y, z = depth_camera.pixel_to_3d(depth_map, u, v)
        
        return {
            'position': (x, y, z),
            'bbox': [x1*scale_x, y1*scale_y, x2*scale_x, y2*scale_y]
        }


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


class URController:
    """
    Classe para controle do robô UR via interface RTDE
    
    Atributos:
        rtde_c (rtde.RTDE): Conexão com o robô
        config (rtde_config.ConfigFile): Configuração da interface
    """
    
    def __init__(self, robot_ip: str, config_file: str = 'rtde_config.xml'):
        """
        Inicializa conexão com o robô UR
        
        Args:
            robot_ip (str): Endereço IP do robô
            config_file (str): Arquivo de configuração RTDE
        """
        self.rtde_c = rtde.RTDE(robot_ip, 30004)
        self.config = rtde_config.ConfigFile(config_file)
        self.setup_connection()
    
    def setup_connection(self):
        """Configura conexão RTDE com o robô"""
        self.rtde_c.connect()
        
        if not self.rtde_c.is_connected():
            raise ConnectionError("Falha ao conectar ao robô UR")
        
        # Configurar recipes para leitura/escrita
        self.state_names, self.state_types = self.config.get_recipe('state')
        self.setp_names, self.setp_types = self.config.get_recipe('setp')
        self.watchdog_names, self.watchdog_types = self.config.get_recipe('watchdog')
        
        # Configurar controle
        self.rtde_c.send_output_setup(self.state_names, self.state_types)
        self.rtde_c.send_input_setup(self.setp_names, self.setp_types)
    
    def get_current_pose(self) -> np.ndarray:
        """Obtém a pose atual do robô"""
        return self.rtde_c.receive().actual_TCP_pose
    
    def move_to_pose(self, pose: np.ndarray, 
                   velocity: float = 0.2, 
                   acceleration: float = 0.5) -> bool:
        """
        Move o robô para uma pose específica
        
        Args:
            pose (np.ndarray): Pose alvo [x, y, z, rx, ry, rz]
            velocity (float): Velocidade normalizada (0-1)
            acceleration (float): Aceleração normalizada (0-1)
            
        Returns:
            bool: True se movimento foi iniciado com sucesso
        """
        # Verificar se pose é válida
        if len(pose) != 6:
            raise ValueError("Pose deve conter 6 valores [x,y,z,rx,ry,rz]")
        
        # Configurar movimento
        self.rtde_c.send(self.setp_names, self.setp_types, pose)
        return True
    
    def generate_pick_script(self, 
                           approach_pose: np.ndarray, 
                           target_pose: np.ndarray, 
                           retreat_pose: np.ndarray) -> str:
        """
        Gera script URScript para sequência de pick-and-place
        
        Args:
            approach_pose (np.ndarray): Pose de aproximação
            target_pose (np.ndarray): Pose para pegar objeto
            retreat_pose (np.ndarray): Pose de retirada
            
        Returns:
            str: Script URScript completo
        """
        script = f"""
def pick_sequence():
    # Abrir gripper
    set_tool_digital_out(0, False)
    sleep(0.5)
    
    # Movimento de aproximação
    movel(p{approach_pose.tolist()}, a={0.5}, v={0.2})
    
    # Movimento para pegar
    movel(p{target_pose.tolist()}, a={0.3}, v={0.1})
    
    # Fechar gripper
    set_tool_digital_out(0, True)
    sleep(0.5)
    
    # Movimento de retirada
    movel(p{retreat_pose.tolist()}, a={0.5}, v={0.2})
end
"""
        return script
    
    def execute_script(self, script: str):
        """Executa script URScript no robô"""
        self.rtde_c.send_program(script)
    
    def disconnect(self):
        """Desconecta do robô"""
        self.rtde_c.disconnect()


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


# Exemplo de uso
if __name__ == "__main__":
    # Configurações
    CONFIG = {
        'camera': {
            'width': 640,
            'height': 480,
            'fps': 30
        },
        'model_path': 'object_detector.h5',
        'calibration_file': 'calibration.json',
        'robot_ip': '192.168.1.10'
    }
    
    # Inicializar sistema
    system = VisionToMotionController(
        camera_config=CONFIG['camera'],
        model_path=CONFIG['model_path'],
        calibration_file=CONFIG['calibration_file'],
        robot_ip=CONFIG['robot_ip']
    )
    
    try:
        # Loop principal
        while True:
            success = system.run_single_cycle()
            print(f"Ciclo {'bem-sucedido' if success else 'sem detecção'}")
            time.sleep(1)  # Intervalo entre ciclos
            
    except KeyboardInterrupt:
        print("Parando sistema...")
    finally:
        system.shutdown()