import tensorflow as tf
import numpy as np
from typing import Optional, Dict

class ObjectDetector:
    # todo: copiar código da sua classe ObjectDetector aqui
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
