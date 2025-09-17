#!/usr/bin/env python3
"""
Sistema de Objetos Virtuais - Python
=====================================
Detecta múltiplos marcadores ArUco e sobrepõe objetos virtuais 3D
(Cubo, Pirâmide, Octaedro)
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any

# --- Dicionário de Formas Geométricas ---
# Define os vértices e faces para cada forma.
# As coordenadas são relativas ao centro e serão escaladas para o tamanho do marcador.

# Cubo
CUBE = {
    "vertices": np.array([
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1], # Frente
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1] # Trás
    ], dtype=np.float32),
    "faces": [
        [0, 1, 2, 3], [1, 5, 6, 2], [5, 4, 7, 6],
        [4, 0, 3, 7], [3, 2, 6, 7], [4, 5, 1, 0]
    ]
}

# Pirâmide de base quadrada
PYRAMID = {
    "vertices": np.array([
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1], # Base
        [0, 0, -1] # Topo
    ], dtype=np.float32),
    "faces": [
        [0, 1, 2, 3], # Base
        [0, 1, 4], # Lado 1
        [1, 2, 4], # Lado 2
        [2, 3, 4], # Lado 3
        [3, 0, 4]  # Lado 4
    ]
}

# Octaedro
OCTAHEDRON = {
    "vertices": np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
    ], dtype=np.float32),
    "faces": [
        [0, 4, 2], [0, 2, 5], [0, 5, 3], [0, 3, 4], # Faces superiores
        [1, 2, 4], [1, 5, 2], [1, 3, 5], [1, 4, 3]  # Faces inferiores
    ]
}

class CuboVirtual:
    """Sistema principal de detecção e rastreamento de objetos virtuais."""
    
    def __init__(self):
        """Inicializa o sistema de cubo virtual."""
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.parameters = cv2.aruco.DetectorParameters()
        
        self.camera_matrix = np.array([
            [800, 0, 320], [0, 800, 240], [0, 0, 1]
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        
        # Mapeamento de ID para forma e cor
        self.mapeamento_objetos = {
            0: {"forma": CUBE, "cor": (0, 255, 0)},       # ID 0 -> Cubo Verde
            1: {"forma": PYRAMID, "cor": (0, 0, 255)},    # ID 1 -> Pirâmide Azul
            2: {"forma": OCTAHEDRON, "cor": (255, 0, 0)}, # ID 2 -> Octaedro Vermelho
            # Adicione mais IDs e formas aqui, se desejar
        }
        
        print("🎯 Sistema de Objetos Virtuais Python inicializado!")
        print("📋 Testando com Cubo (ID=0), Pirâmide (ID=1) e Octaedro (ID=2).")
        print("🔗 Link: https://chev.me/arucogen/ (selecione 6x6)")
    
    def detectar_e_desenhar_objetos(self, frame: np.ndarray) -> bool:
        """
        Detecta todos os marcadores e desenha objetos correspondentes.
        """
        corners, ids, _ = cv2.aruco.detectMarkers(
            frame, self.aruco_dict, parameters=self.parameters
        )
        
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            for i, detected_id in enumerate(ids):
                marker_id = detected_id[0]
                
                # Se o ID do marcador estiver no nosso mapeamento
                if marker_id in self.mapeamento_objetos:
                    # Obter o tamanho do marcador dinamicamente (largura média dos cantos)
                    marker_size_in_meters = np.mean(np.linalg.norm(corners[i][0][0] - corners[i][0][1])) / 1000.0
                    
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners[i:i+1], marker_size_in_meters, 
                        self.camera_matrix, self.dist_coeffs
                    )
                    
                    objeto_info = self.mapeamento_objetos[marker_id]
                    forma = objeto_info["forma"]
                    cor = objeto_info["cor"]

                    self._desenhar_eixos_3d(frame, rvecs[0][0], tvecs[0][0], marker_size_in_meters)
                    self._desenhar_objeto_virtual(frame, rvecs[0][0], tvecs[0][0], forma, cor, marker_size_in_meters)
            
            return True
        
        return False

    def _desenhar_objeto_virtual(self, frame: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, 
                                 forma: Dict, cor: Tuple, tamanho_marcador: float) -> None:
        """Desenha um objeto virtual 3D sólido."""
        
        # Escalar os vértices da forma para o tamanho do marcador
        scaled_vertices = forma["vertices"] * (tamanho_marcador / 2.0)
        faces_indices = forma["faces"]

        img_pts, _ = cv2.projectPoints(scaled_vertices, rvec, tvec, 
                                     self.camera_matrix, self.dist_coeffs)
        img_pts = np.int32(img_pts).reshape(-1, 2)
        
        distancias_z = [np.mean([tvec[2] + scaled_vertices[i][2] for i in face]) for face in faces_indices]
        faces_ordenadas = sorted(zip(faces_indices, distancias_z), key=lambda x: x[1], reverse=True)

        for face, _ in faces_ordenadas:
            pontos_face = np.array([img_pts[i] for i in face])
            cv2.fillConvexPoly(frame, pontos_face, cor)
        
        for face in faces_indices:
            pontos_face = np.array([img_pts[i] for i in face])
            cv2.polylines(frame, [pontos_face], True, (0, 0, 0), 1)

    def _desenhar_eixos_3d(self, frame, rvec, tvec, tamanho_marcador):
        """Desenha eixos 3D que se ajustam ao tamanho do marcador."""
        axis_length = tamanho_marcador * 0.6
        axis_pts = np.array([
            [0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, -axis_length]
        ], dtype=np.float32)
        
        img_pts, _ = cv2.projectPoints(axis_pts, rvec, tvec, 
                                    self.camera_matrix, self.dist_coeffs)
        img_pts = img_pts.astype(int)
        
        origem = tuple(img_pts[0][0])
        cv2.arrowedLine(frame, origem, tuple(img_pts[1][0]), (0, 0, 255), 5)  # X=vermelho
        cv2.arrowedLine(frame, origem, tuple(img_pts[2][0]), (0, 255, 0), 5)  # Y=verde
        cv2.arrowedLine(frame, origem, tuple(img_pts[3][0]), (255, 0, 0), 5)  # Z=azul

def main():
    """Função principal do sistema."""
    print("🎯 === SISTEMA DE OBJETOS VIRTUAIS PYTHON ===")
    
    try:
        sistema = CuboVirtual()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("⚠️  Câmera não disponível.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n🎮 CONTROLES:")
        print("   ESC = Sair")
        print("   Pressione qualquer tecla para continuar...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            sistema.detectar_e_desenhar_objetos(frame)
            
            cv2.imshow("🎯 Sistema de Multiplos Objetos Virtuais", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n🎯 Sistema finalizado!")
        
    except KeyboardInterrupt:
        print("\n🛑 Sistema interrompido pelo usuário")
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()