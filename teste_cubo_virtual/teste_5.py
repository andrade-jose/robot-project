#!/usr/bin/env python3
"""
Sistema de Cubo Virtual - Python
=================================
Detecta múltiplos marcadores ArUco e sobrepõe objetos virtuais 3D
"""

import cv2
import numpy as np
import time
import json
import os
from typing import List, Tuple, Optional, Dict, Any

class CuboVirtual:
    """Sistema principal de detecção e rastreamento de objetos virtuais."""
    
    def __init__(self):
        """Inicializa o sistema de cubo virtual."""
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.parameters = cv2.aruco.DetectorParameters()
        
        self.camera_matrix = np.array([
            [800, 0, 320],    # fx, 0, cx
            [0, 800, 240],    # 0, fy, cy
            [0, 0, 1]         # 0, 0, 1
        ], dtype=np.float64)
        
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        
        self.objetos: Dict[int, Dict[str, Any]] = {}
        
        print("🎯 Sistema de Cubo Virtual Python inicializado!")
        print("📋 Para testar: imprima vários marcadores ArUco (ex: ID=0, 1, 2)")
        print("🔗 Link: https://chev.me/arucogen/ (selecione 6x6)")
    
    def detectar_e_desenhar_objetos(self, frame: np.ndarray) -> bool:
        """
        Detecta todos os marcadores ArUco e desenha objetos correspondentes.
        """
        corners, ids, _ = cv2.aruco.detectMarkers(
            frame, self.aruco_dict, parameters=self.parameters
        )
        
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            visiveis_agora = [id[0] for id in ids]
            self.objetos = {id: obj for id, obj in self.objetos.items() if id in visiveis_agora}

            for i, detected_id in enumerate(ids):
                marker_id = detected_id[0]
                
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners[i:i+1], 0.05, self.camera_matrix, self.dist_coeffs
                )
                
                cor_objeto = self._definir_cor_por_id(marker_id)

                self._desenhar_eixos_3d(frame, rvecs[0][0], tvecs[0][0])
                self._desenhar_cubo_virtual(frame, rvecs[0][0], tvecs[0][0], cor_objeto)
                
                self.objetos[marker_id] = {
                    'posicao': tvecs[0][0],
                    'rotacao': rvecs[0][0],
                    'visivel': True,
                    'timestamp': time.time(),
                    'id': marker_id
                }
            return True
        
        self.objetos.clear()
        return False
    
    def _definir_cor_por_id(self, id: int) -> Tuple[int, int, int]:
        cores_base = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]
        return cores_base[id % len(cores_base)]

    def _desenhar_cubo_virtual(self, frame: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, cor: Tuple) -> None:
        """Desenha cubo virtual 3D sólido com cor específica."""
        cube_pts = np.array([
            [-0.025, -0.025, 0.025], [0.025, -0.025, 0.025], [0.025, 0.025, 0.025], [-0.025, 0.025, 0.025],
            [-0.025, -0.025, -0.025], [0.025, -0.025, -0.025], [0.025, 0.025, -0.025], [-0.025, 0.025, -0.025]
        ], dtype=np.float32)

        faces_indices = [
            [0, 1, 2, 3], [1, 5, 6, 2], [5, 4, 7, 6],
            [4, 0, 3, 7], [3, 2, 6, 7], [4, 5, 1, 0]
        ]
        
        img_pts, _ = cv2.projectPoints(cube_pts, rvec, tvec, 
                                     self.camera_matrix, self.dist_coeffs)
        img_pts = np.int32(img_pts).reshape(-1, 2)
        
        distancias_z = [np.mean([tvec[2] + cube_pts[i][2] for i in face]) for face in faces_indices]
        faces_ordenadas = sorted(zip(faces_indices, distancias_z), key=lambda x: x[1], reverse=True)

        for face, _ in faces_ordenadas:
            pontos_face = np.array([img_pts[i] for i in face])
            cv2.fillConvexPoly(frame, pontos_face, cor)
        
        for face in faces_indices:
            pontos_face = np.array([img_pts[i] for i in face])
            cv2.polylines(frame, [pontos_face], True, (0, 0, 0), 1)

    def _desenhar_eixos_3d(self, frame, rvec, tvec):
        """Desenha eixos 3D compatível com OpenCV 4.12.0"""
        axis_pts = np.array([
            [0, 0, 0], [0.03, 0, 0], [0, 0.03, 0], [0, 0, -0.03]
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
    print("🎯 === SISTEMA CUBO VIRTUAL PYTHON ===")
    print(f"📹 OpenCV Version: {cv2.__version__}")
    
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