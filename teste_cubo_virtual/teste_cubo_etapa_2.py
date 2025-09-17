#!/usr/bin/env python3
"""
Sistema de Cubo Virtual em Realidade Aumentada - Versão Limpa
============================================================
Detecta marcadores de referência (0,1) e cria cubo virtual verde
Detecta grupos de marcadores:
- Grupo 1 (2,4,6): Cubos vermelhos
- Grupo 2 (3,5,7): Cubos azuis
"""

import cv2
import numpy as np
import time
from typing import Dict, Any

class CuboVirtualAR:
    """Sistema limpo de cubo virtual em realidade aumentada."""
    
    def __init__(self, altura_fixa_mm: int = 80):
        """Inicializa o sistema."""
        # Configurar ArUco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.tamanho_marcador_metros = 0.03  # 3cm - ajustar conforme necessário
        
        # Parâmetros da câmera
        self.camera_matrix = np.array([
            [800, 0, 320],
            [0, 800, 240],
            [0, 0, 1]
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        
        # Configuração dos marcadores
        self.altura_fixa_mm = altura_fixa_mm
        self.grupo_1 = [2, 4, 6]  # Cubos vermelhos
        self.grupo_2 = [3, 5, 7]  # Cubos azuis
        
        # Variáveis de estado
        self.marcadores_ref = {}  # IDs 0 e 1
        self.marcadores_g1 = {}   # IDs 2, 4, 6
        self.marcadores_g2 = {}   # IDs 3, 5, 7
        
        self.sistema_calibrado = False
        self.origem_3d = None
        self.vetor_x = None
        self.vetor_y = None
        self.distancia_refs_mm = 0
        
        print("Sistema Cubo Virtual AR inicializado")
        print(f"Altura do cubo: {altura_fixa_mm}mm")
        print(f"Grupo 1 {self.grupo_1} = Cubos VERMELHOS")
        print(f"Grupo 2 {self.grupo_2} = Cubos AZUIS")

    def detectar_marcadores(self, frame: np.ndarray) -> bool:
        """Detecta todos os marcadores e classifica por grupos."""
        # Limpar detecções anteriores
        self.marcadores_ref.clear()
        self.marcadores_g1.clear()
        self.marcadores_g2.clear()
        
        # Detectar marcadores ArUco
        corners, ids, _ = cv2.aruco.detectMarkers(frame, self.aruco_dict, parameters=self.parameters)
        
        if ids is not None and len(ids) > 0:
            # Desenhar marcadores detectados
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Processar cada marcador
            for i, marker_id in enumerate(ids):
                marker_id = marker_id[0]
                self._processar_marcador(marker_id, corners[i])
            
            # Calibrar sistema se temos marcadores de referência
            if len(self.marcadores_ref) == 2:
                self._calibrar_sistema()
            
            return True
        
        return False
    
    def _processar_marcador(self, marker_id: int, corners: np.ndarray):
        """Processa um marcador detectado e classifica por grupo."""
        # Estimar pose do marcador
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            [corners], self.tamanho_marcador_metros,
            self.camera_matrix, self.dist_coeffs
        )
        
        # Criar info do marcador
        info_marcador = {
            'id': marker_id,
            'posicao': tvecs[0][0],
            'rotacao': rvecs[0][0],
            'corners': corners,
            'timestamp': time.time()
        }
        
        # Classificar por grupo
        if marker_id in [0, 1]:
            self.marcadores_ref[marker_id] = info_marcador
        elif marker_id in self.grupo_1:
            self.marcadores_g1[marker_id] = info_marcador
        elif marker_id in self.grupo_2:
            self.marcadores_g2[marker_id] = info_marcador
    
    def _calibrar_sistema(self):
        """Calibra o sistema de coordenadas usando marcadores 0 e 1."""
        if 0 not in self.marcadores_ref or 1 not in self.marcadores_ref:
            return False
        
        # Posições dos marcadores de referência
        pos_0 = self.marcadores_ref[0]['posicao']
        pos_1 = self.marcadores_ref[1]['posicao']
        
        # Definir origem e vetores
        self.origem_3d = pos_0.copy()
        
        # Vetor X (de 0 para 1)
        vetor_x_bruto = pos_1 - pos_0
        self.vetor_x = vetor_x_bruto / np.linalg.norm(vetor_x_bruto)
        
        # Vetor Y (perpendicular)
        vetor_z = np.array([0, 0, 1])
        self.vetor_y = np.cross(vetor_z, self.vetor_x)
        self.vetor_y = self.vetor_y / np.linalg.norm(self.vetor_y)
        
        # Calcular distância
        self.distancia_refs_mm = np.linalg.norm(vetor_x_bruto) * 1000
        
        self.sistema_calibrado = True
        return True
    
    def desenhar_realidade_aumentada(self, frame: np.ndarray):
        """Desenha todos os elementos de realidade aumentada."""
        # Verificar se ainda temos os marcadores de referência
        if len(self.marcadores_ref) == 2 and 0 in self.marcadores_ref and 1 in self.marcadores_ref:
            # Se perdemos a calibração, recalibrar
            if not self.sistema_calibrado:
                self._calibrar_sistema()
            
            # Desenhar cubo principal
            if self.sistema_calibrado:
                self._desenhar_cubo_principal(frame)
        else:
            # Se não temos os 2 marcadores, perder calibração
            self.sistema_calibrado = False
        
        # Cubos pequenos dos grupos (sempre desenhar se detectados)
        self._desenhar_cubos_grupos(frame)
        
        # Informações na tela
        self._desenhar_informacoes(frame)
    
    def _desenhar_cubo_principal(self, frame: np.ndarray):
        """Desenha o cubo virtual principal verde."""
        # Centro entre os marcadores
        pos_0 = self.marcadores_ref[0]['posicao']
        pos_1 = self.marcadores_ref[1]['posicao']
        centro = (pos_0 + pos_1) / 2
        
        # Dimensões do cubo
        largura = self.distancia_refs_mm / 1000.0  # Converter para metros
        altura = largura  # Quadrado
        profundidade = self.altura_fixa_mm / 1000.0
        
        # Vértices do cubo
        half_w = largura / 2
        half_h = altura / 2
        
        vertices_locais = np.array([
            [-half_w, -half_h, 0], [half_w, -half_h, 0],
            [half_w, half_h, 0], [-half_w, half_h, 0],
            [-half_w, -half_h, profundidade], [half_w, -half_h, profundidade],
            [half_w, half_h, profundidade], [-half_w, half_h, profundidade]
        ], dtype=np.float32)
        
        # Transformar para coordenadas do mundo
        vertices_mundo = []
        for v in vertices_locais:
            v_mundo = centro + v[0] * self.vetor_x + v[1] * self.vetor_y + np.array([0, 0, v[2]])
            vertices_mundo.append(v_mundo)
        
        # Projetar para 2D
        vertices_mundo = np.array(vertices_mundo, dtype=np.float32)
        img_pts, _ = cv2.projectPoints(
            vertices_mundo, np.zeros(3), np.zeros(3),
            self.camera_matrix, self.dist_coeffs
        )
        img_pts = img_pts.astype(int)
        
        # Desenhar arestas do cubo
        arestas = [
            # Base
            (0,1), (1,2), (2,3), (3,0),
            # Topo
            (4,5), (5,6), (6,7), (7,4),
            # Verticais
            (0,4), (1,5), (2,6), (3,7)
        ]
        
        for inicio, fim in arestas:
            pt1 = tuple(img_pts[inicio][0])
            pt2 = tuple(img_pts[fim][0])
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)  # Verde
        
        # Destacar base
        base_arestas = [(0,1), (1,2), (2,3), (3,0)]
        for inicio, fim in base_arestas:
            pt1 = tuple(img_pts[inicio][0])
            pt2 = tuple(img_pts[fim][0])
            cv2.line(frame, pt1, pt2, (0, 255, 255), 3)  # Amarelo
    
    def _desenhar_cubos_grupos(self, frame: np.ndarray):
        """Desenha cubos pequenos nos marcadores dos grupos."""
        # Grupo 1 - Vermelho
        for info in self.marcadores_g1.values():
            self._desenhar_cubo_pequeno(frame, info, (0, 0, 255), "G1")
        
        # Grupo 2 - Azul
        for info in self.marcadores_g2.values():
            self._desenhar_cubo_pequeno(frame, info, (255, 0, 0), "G2")
    
    def _desenhar_cubo_pequeno(self, frame: np.ndarray, info: dict, cor: tuple, grupo: str):
        """Desenha um cubo pequeno sobre um marcador."""
        rvec = info['rotacao']
        tvec = info['posicao']
        marker_id = info['id']
        
        # Tamanho do cubo pequeno
        tamanho = self.tamanho_marcador_metros * 0.7
        
        # Vértices do cubo pequeno
        vertices = np.array([
            [0, 0, 0], [tamanho, 0, 0], [tamanho, tamanho, 0], [0, tamanho, 0],
            [0, 0, -tamanho], [tamanho, 0, -tamanho], 
            [tamanho, tamanho, -tamanho], [0, tamanho, -tamanho]
        ], dtype=np.float32)
        
        # Projetar para 2D
        img_pts, _ = cv2.projectPoints(vertices, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        img_pts = img_pts.astype(int)
        
        # Desenhar arestas
        arestas = [
            (0,1), (1,2), (2,3), (3,0),  # Base
            (4,5), (5,6), (6,7), (7,4),  # Topo
            (0,4), (1,5), (2,6), (3,7)   # Verticais
        ]
        
        for inicio, fim in arestas:
            pt1 = tuple(img_pts[inicio][0])
            pt2 = tuple(img_pts[fim][0])
            cv2.line(frame, pt1, pt2, cor, 2)
        
        # Destacar marcador
        corners = info['corners']
        centro = np.mean(corners[0], axis=0).astype(int)
        cv2.circle(frame, tuple(centro), 20, cor, 2)
        cv2.putText(frame, f"{grupo}({marker_id})", 
                   (centro[0] + 25, centro[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)
    
    def _desenhar_informacoes(self, frame: np.ndarray):
        """Desenha informações na tela."""
        y = 30
        
        # Status do cubo principal
        if self.sistema_calibrado:
            cv2.putText(frame, "CUBO VIRTUAL ATIVO", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y += 25
            cv2.putText(frame, f"Distancia refs: {self.distancia_refs_mm:.1f}mm", 
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 20
        else:
            refs_detectadas = len(self.marcadores_ref)
            cv2.putText(frame, f"AGUARDANDO REFS {refs_detectadas}/2", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y += 25
        
        # Informações dos grupos
        cv2.putText(frame, f"Grupo 1: {len(self.marcadores_g1)} marcadores", 
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        y += 18
        cv2.putText(frame, f"Grupo 2: {len(self.marcadores_g2)} marcadores", 
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        y += 18
        
        # IDs detectados
        if self.marcadores_g1:
            ids_g1 = list(self.marcadores_g1.keys())
            cv2.putText(frame, f"IDs G1: {ids_g1}", 
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            y += 15
        
        if self.marcadores_g2:
            ids_g2 = list(self.marcadores_g2.keys())
            cv2.putText(frame, f"IDs G2: {ids_g2}", 
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    def imprimir_info_detalhada(self):
        """Imprime informações detalhadas."""
        print("\n=== INFORMAÇÕES DETALHADAS ===")
        
        # Cubo principal
        if self.sistema_calibrado:
            print(f"CUBO PRINCIPAL (Verde): ATIVO")
            print(f"  Distância entre refs: {self.distancia_refs_mm:.1f}mm")
            print(f"  Dimensões: {self.distancia_refs_mm:.1f}x{self.distancia_refs_mm:.1f}x{self.altura_fixa_mm}mm")
        else:
            print("CUBO PRINCIPAL: INATIVO (necessário IDs 0 e 1)")
        
        # Grupo 1
        print(f"\nGRUPO 1 (Vermelho): {len(self.marcadores_g1)} detectados")
        for marker_id, info in self.marcadores_g1.items():
            pos = info['posicao'] * 1000
            print(f"  ID {marker_id}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})mm")
        
        # Grupo 2
        print(f"\nGRUPO 2 (Azul): {len(self.marcadores_g2)} detectados")
        for marker_id, info in self.marcadores_g2.items():
            pos = info['posicao'] * 1000
            print(f"  ID {marker_id}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})mm")

def main():
    """Função principal."""
    print("=== SISTEMA CUBO VIRTUAL AR ===")
    print("Marcadores necessários:")
    print("- IDs 0 e 1: Marcadores de referência (cubo verde)")
    print("- IDs 2,4,6: Grupo 1 (cubos vermelhos)")
    print("- IDs 3,5,7: Grupo 2 (cubos azuis)")
    
    sistema = CuboVirtualAR(altura_fixa_mm=80)
    
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Erro: Câmera não disponível")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\nControles:")
    print("ESC = Sair")
    print("I = Informações detalhadas")
    print("ESPAÇO = Screenshot")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detectar marcadores
        sistema.detectar_marcadores(frame)
        
        # Desenhar realidade aumentada
        sistema.desenhar_realidade_aumentada(frame)
        
        cv2.imshow("Sistema Cubo Virtual AR", frame)
        
        # Controles
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('i'):
            sistema.imprimir_info_detalhada()
        elif key == ord(' '):
            filename = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Screenshot salvo: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Sistema finalizado")

if __name__ == "__main__":
    main()