import numpy as np
import pyrealsense2 as rs
from typing import Tuple, Dict

class DepthCamera:
    # todo: copiar código da sua classe DepthCamera aqui
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
