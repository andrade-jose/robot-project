#!/usr/bin/env python3
"""
Etapa 1 - Cubo Virtual com Realidade Aumentada
===============================================
Detecta 2 marcadores de referência, mede distância entre eles,
cria cubo/retângulo virtual e mostra em realidade aumentada
"""

import cv2
import numpy as np
import time
from typing import Dict, Any, Optional, Tuple

class CuboVirtualAR:
    """Sistema de cubo virtual em realidade aumentada - Etapa 1."""
    
    def __init__(self, altura_fixa_mm: int = 100):
        """
        Inicializa o sistema de cubo virtual AR.
        
        Args:
            altura_fixa_mm: Altura do cubo virtual em mm
        """
        self._configurar_aruco()
        self._configurar_camera()
        self.altura_fixa_mm = altura_fixa_mm
        self._inicializar_variaveis()
        self._imprimir_configuracao()
    
    def _configurar_aruco(self):
        """Configura detector ArUco."""
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.tamanho_marcador_metros = 0.03  # 3cm - ajuste conforme seu marcador
    
    def _configurar_camera(self):
        """Configura parâmetros da câmera."""
        self.camera_matrix = np.array([
            [800, 0, 320],
            [0, 800, 240],
            [0, 0, 1]
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    
    def _inicializar_variaveis(self):
        """Inicializa variáveis de estado."""
        # IDs dos marcadores de referência
        self.id_origem = 0    # Primeiro marcador
        self.id_eixo_x = 1    # Segundo marcador
        
        # Dados dos marcadores
        self.marcadores_referencia: Dict[int, Dict[str, Any]] = {}
        
        # Sistema de coordenadas
        self.sistema_calibrado = False
        self.origem_3d = None
        self.vetor_x = None
        self.vetor_y = None
        self.largura_cubo_mm = 0
        self.altura_cubo_mm = 0  # Será igual à largura para formar um quadrado
        
        # Informações da medição
        self.distancia_marcadores_mm = 0
        self.info_medicao = {
            'distancia_real': 0,
            'largura_cubo': 0,
            'altura_cubo': 0,
            'posicao_centro': None,
            'ultima_medicao': 0
        }
    
    def _imprimir_configuracao(self):
        """Imprime configuração inicial."""
        print("=== ETAPA 1: CUBO VIRTUAL EM REALIDADE AUMENTADA ===")
        print(f"Altura do cubo: {self.altura_fixa_mm}mm (fixa)")
        print("Marcadores necessários:")
        print(f"  ID {self.id_origem} = Primeiro marcador")
        print(f"  ID {self.id_eixo_x} = Segundo marcador")
        print("O cubo será criado baseado na distância entre os marcadores")

    def detectar_e_medir(self, frame: np.ndarray) -> bool:
        """Detecta marcadores e mede distância para criar cubo virtual."""
        self.marcadores_referencia.clear()
        
        corners, ids, _ = cv2.aruco.detectMarkers(
            frame, self.aruco_dict, parameters=self.parameters
        )
        
        if ids is not None and len(ids) > 0:
            # Desenhar marcadores detectados
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            for i, marker_id in enumerate(ids):
                marker_id = marker_id[0]
                
                # Processar apenas os marcadores de referência
                if marker_id in [self.id_origem, self.id_eixo_x]:
                    self._processar_marcador_referencia(marker_id, corners[i])
            
            # Se temos os 2 marcadores, calibrar e desenhar cubo
            if len(self.marcadores_referencia) == 2:
                if self._calibrar_sistema():
                    self._desenhar_cubo_virtual_ar(frame)
                    self._mostrar_informacoes_tela(frame)
                    return True
        
        return False
    
    def _processar_marcador_referencia(self, marker_id: int, corners: np.ndarray):
        """Processa um marcador de referência."""
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            [corners], self.tamanho_marcador_metros,
            self.camera_matrix, self.dist_coeffs
        )
        
        self.marcadores_referencia[marker_id] = {
            'id': marker_id,
            'posicao_mundo': tvecs[0][0],
            'rotacao': rvecs[0][0],
            'corners': corners,
            'timestamp': time.time()
        }
    
    def _calibrar_sistema(self) -> bool:
        """Calibra o sistema e calcula dimensões do cubo."""
        if (self.id_origem not in self.marcadores_referencia or
            self.id_eixo_x not in self.marcadores_referencia):
            return False
        
        # Obter posições dos marcadores
        pos_origem = self.marcadores_referencia[self.id_origem]['posicao_mundo']
        pos_x = self.marcadores_referencia[self.id_eixo_x]['posicao_mundo']
        
        # Configurar sistema de coordenadas
        self.origem_3d = pos_origem.copy()
        
        # Calcular vetor X (direção entre os marcadores)
        vetor_x_bruto = pos_x - pos_origem
        self.vetor_x = vetor_x_bruto / np.linalg.norm(vetor_x_bruto)
        
        # Calcular vetor Y (perpendicular no plano horizontal)
        vetor_z_mundo = np.array([0, 0, 1])  # Vertical
        self.vetor_y = np.cross(vetor_z_mundo, self.vetor_x)
        self.vetor_y = self.vetor_y / np.linalg.norm(self.vetor_y)
        
        # Calcular distância real entre marcadores
        self.distancia_marcadores_mm = np.linalg.norm(vetor_x_bruto) * 1000
        
        # Definir dimensões do cubo baseado na distância
        self.largura_cubo_mm = self.distancia_marcadores_mm  # Largura = distância entre marcadores
        self.altura_cubo_mm = self.distancia_marcadores_mm   # Altura = mesma distância (quadrado)
        
        # Calcular centro do cubo (meio entre os marcadores)
        centro_mundo = (pos_origem + pos_x) / 2
        
        # Atualizar informações
        self.info_medicao = {
            'distancia_real': self.distancia_marcadores_mm,
            'largura_cubo': self.largura_cubo_mm,
            'altura_cubo': self.altura_cubo_mm,
            'posicao_centro': centro_mundo,
            'ultima_medicao': time.time()
        }
        
        self.sistema_calibrado = True
        return True
    
    def _desenhar_cubo_virtual_ar(self, frame: np.ndarray):
        """Desenha o cubo virtual em realidade aumentada."""
        if not self.sistema_calibrado:
            return
        
        # Calcular centro do cubo
        centro_mundo = self.info_medicao['posicao_centro']
        
        # Definir metade das dimensões
        half_w = self.largura_cubo_mm / 2000.0  # Converter mm para metros e dividir por 2
        half_h = self.altura_cubo_mm / 2000.0
        half_z = self.altura_fixa_mm / 2000.0
        
        # Vértices do cubo centrado entre os marcadores
        vertices_locais = np.array([
            [-half_w, -half_h, 0],      [half_w, -half_h, 0],      # Base frontal
            [half_w, half_h, 0],        [-half_w, half_h, 0],      # Base traseira
            [-half_w, -half_h, half_z*2], [half_w, -half_h, half_z*2], # Topo frontal
            [half_w, half_h, half_z*2],   [-half_w, half_h, half_z*2]   # Topo traseiro
        ], dtype=np.float32)
        
        # Transformar vértices para o sistema de coordenadas do mundo
        vertices_mundo = []
        for vertice_local in vertices_locais:
            # Aplicar rotação e translação
            vertice_mundo = (centro_mundo + 
                           vertice_local[0] * self.vetor_x +
                           vertice_local[1] * self.vetor_y +
                           np.array([0, 0, vertice_local[2]]))
            vertices_mundo.append(vertice_mundo)
        
        vertices_mundo = np.array(vertices_mundo, dtype=np.float32)
        
        # Projetar vértices 3D para 2D
        img_pts, _ = cv2.projectPoints(
            vertices_mundo, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]),
            self.camera_matrix, self.dist_coeffs
        )
        img_pts = img_pts.astype(int)
        
        # Desenhar arestas do cubo
        self._desenhar_arestas_cubo(frame, img_pts)
        
        # Desenhar informações adicionais
        self._desenhar_eixos_referencia(frame)
        self._destacar_marcadores_referencia(frame)
    
    def _desenhar_arestas_cubo(self, frame: np.ndarray, img_pts: np.ndarray):
        """Desenha as arestas do cubo virtual."""
        # Definir arestas do cubo (conectando vértices)
        arestas = [
            # Base (vértices 0-3)
            (0, 1), (1, 2), (2, 3), (3, 0),
            # Topo (vértices 4-7) 
            (4, 5), (5, 6), (6, 7), (7, 4),
            # Conexões verticais
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        
        # Desenhar cada aresta
        for inicio, fim in arestas:
            pt1 = tuple(img_pts[inicio][0])
            pt2 = tuple(img_pts[fim][0])
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)  # Verde
        
        # Desenhar base com cor diferente para destacar
        base_arestas = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for inicio, fim in base_arestas:
            pt1 = tuple(img_pts[inicio][0])
            pt2 = tuple(img_pts[fim][0])
            cv2.line(frame, pt1, pt2, (255, 0, 0), 3)  # Azul mais espesso
    
    def _desenhar_eixos_referencia(self, frame: np.ndarray):
        """Desenha eixos 3D nos marcadores de referência."""
        for marker_id, info in self.marcadores_referencia.items():
            rvec = info['rotacao']
            tvec = info['posicao_mundo']
            
            # Pontos dos eixos
            axis_pts = np.array([
                [0, 0, 0], [0.03, 0, 0], [0, 0.03, 0], [0, 0, -0.03]
            ], dtype=np.float32)
            
            img_pts, _ = cv2.projectPoints(
                axis_pts, rvec, tvec, self.camera_matrix, self.dist_coeffs
            )
            img_pts = img_pts.astype(int)
            
            origem = tuple(img_pts[0][0])
            # X = vermelho, Y = verde, Z = azul
            cv2.arrowedLine(frame, origem, tuple(img_pts[1][0]), (0, 0, 255), 3)
            cv2.arrowedLine(frame, origem, tuple(img_pts[2][0]), (0, 255, 0), 3)
            cv2.arrowedLine(frame, origem, tuple(img_pts[3][0]), (255, 0, 0), 3)
    
    def _destacar_marcadores_referencia(self, frame: np.ndarray):
        """Destaca os marcadores de referência."""
        for marker_id, info in self.marcadores_referencia.items():
            corners = info['corners']
            centro = np.mean(corners[0], axis=0).astype(int)
            
            # Círculo colorido
            cor = (0, 0, 255) if marker_id == self.id_origem else (0, 255, 0)
            cv2.circle(frame, tuple(centro), 25, cor, 3)
            
            # Texto do ID
            nome = "ORIGEM" if marker_id == self.id_origem else "EIXO X"
            cv2.putText(frame, f"{nome} (ID{marker_id})", 
                       (centro[0] + 30, centro[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)
    
    def _mostrar_informacoes_tela(self, frame: np.ndarray):
        """Mostra informações do sistema na tela."""
        info = self.info_medicao
        altura_texto = 30
        
        # Status
        cv2.putText(frame, "CUBO VIRTUAL ATIVO", (10, altura_texto), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        altura_texto += 30
        
        # Distância entre marcadores
        cv2.putText(frame, f"Distancia marcadores: {info['distancia_real']:.1f}mm", 
                   (10, altura_texto), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        altura_texto += 25
        
        # Dimensões do cubo
        cv2.putText(frame, f"Cubo: {info['largura_cubo']:.1f}x{info['altura_cubo']:.1f}x{self.altura_fixa_mm}mm", 
                   (10, altura_texto), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        altura_texto += 25
        
        # Forma do cubo
        if abs(info['largura_cubo'] - info['altura_cubo']) < 1:
            forma = "QUADRADO"
        else:
            forma = "RETANGULO"
        cv2.putText(frame, f"Formato base: {forma}", 
                   (10, altura_texto), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    def imprimir_informacoes_detalhadas(self):
        """Imprime informações detalhadas do sistema."""
        if not self.sistema_calibrado:
            print("Sistema não calibrado. Detecte os 2 marcadores de referência.")
            return
        
        info = self.info_medicao
        print("\n=== INFORMAÇÕES DO CUBO VIRTUAL ===")
        print(f"Distância entre marcadores: {info['distancia_real']:.2f}mm")
        print(f"Dimensões do cubo:")
        print(f"  Largura (X): {info['largura_cubo']:.2f}mm")
        print(f"  Altura (Y): {info['altura_cubo']:.2f}mm")
        print(f"  Profundidade (Z): {self.altura_fixa_mm}mm")
        
        if abs(info['largura_cubo'] - info['altura_cubo']) < 1:
            print("Formato da base: QUADRADO")
        else:
            print("Formato da base: RETÂNGULO")
        
        print(f"Centro do cubo: {info['posicao_centro']}")
        print(f"Última medição: {time.ctime(info['ultima_medicao'])}")

def main():
    """Função principal - Etapa 1."""
    print("=== ETAPA 1: CUBO VIRTUAL EM REALIDADE AUMENTADA ===")
    print("Instruções:")
    print("1. Coloque o marcador ID 0 em uma posição")
    print("2. Coloque o marcador ID 1 em outra posição")
    print("3. O sistema criará um cubo virtual baseado na distância entre eles")
    print("4. Você verá o cubo em realidade aumentada na tela")
    
    try:
        # Inicializar sistema
        sistema = CuboVirtualAR(altura_fixa_mm=80)  # Cubo com 8cm de altura
        
        # Abrir câmera
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Erro: Câmera não disponível.")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\nCONTROLES:")
        print("ESC = Sair")
        print("I = Mostrar informações detalhadas")
        print("ESPAÇO = Capturar screenshot")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detectar marcadores e desenhar cubo virtual
            cubo_ativo = sistema.detectar_e_medir(frame)
            
            # Mostrar status se cubo não estiver ativo
            if not cubo_ativo:
                refs_detectadas = len(sistema.marcadores_referencia)
                cv2.putText(frame, f"DETECTANDO MARCADORES... {refs_detectadas}/2", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Necessario: ID 0 e ID 1", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
            
            cv2.imshow("Etapa 1 - Cubo Virtual AR", frame)
            
            # Processar teclas
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord('i') or key == ord('I'):
                sistema.imprimir_informacoes_detalhadas()
            elif key == ord(' '):  # ESPAÇO
                timestamp = int(time.time())
                filename = f"cubo_virtual_ar_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot salvo: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("Etapa 1 concluída!")
        
    except KeyboardInterrupt:
        print("Sistema interrompido pelo usuário.")
    except Exception as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    main()