#!/usr/bin/env python3
"""
Sistema de Matriz 3D com Marcadores de Referência
===============================================
Usa 2 marcadores fixos para criar uma matriz 3D grande do tabuleiro
Detecta peças com marcadores e fornece coordenadas precisas para garra robótica
"""

import cv2
import numpy as np
import time
import json
import math
from typing import Dict, List, Tuple, Any, Optional

class SistemaMatrizTabuleiro:
    """Sistema de matriz 3D baseado em marcadores de referência fixos."""
    
    def __init__(self, largura_tabuleiro_mm: int = 400, altura_tabuleiro_mm: int = 300, altura_trabalho_mm: int = 100):
        """
        Inicializa o sistema de matriz do tabuleiro.
        
        Args:
            largura_tabuleiro_mm: Largura do tabuleiro em mm
            altura_tabuleiro_mm: Altura do tabuleiro em mm  
            altura_trabalho_mm: Altura de trabalho da garra em mm
        """
        # Configurar detector ArUco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        
        # Parâmetros da câmera (CALIBRAR para sua câmera!)
        self.camera_matrix = np.array([
            [800, 0, 320],
            [0, 800, 240],
            [0, 0, 1]
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        
        # Tamanho físico dos marcadores
        self.tamanho_marcador_metros = 0.03  # 3cm
        
        # Configurações do tabuleiro
        self.largura_tabuleiro_mm = largura_tabuleiro_mm
        self.altura_tabuleiro_mm = altura_tabuleiro_mm
        self.altura_trabalho_mm = altura_trabalho_mm
        
        # Resolução da matriz (pontos por mm)
        self.resolucao_mm = 10  # 1 ponto a cada 10mm
        self.pontos_x = self.largura_tabuleiro_mm // self.resolucao_mm + 1
        self.pontos_y = self.altura_tabuleiro_mm // self.resolucao_mm + 1
        self.pontos_z = self.altura_trabalho_mm // self.resolucao_mm + 1
        
        # IDs dos marcadores de referência (FIXOS no tabuleiro)
        self.id_marcador_ref1 = 0  # Canto inferior esquerdo
        self.id_marcador_ref2 = 1  # Canto inferior direito
        
        # Dados dos marcadores
        self.marcadores_referencia: Dict[int, Dict[str, Any]] = {}
        self.marcadores_pecas: Dict[int, Dict[str, Any]] = {}
        
        # Sistema de coordenadas do tabuleiro
        self.sistema_calibrado = False
        self.origem_tabuleiro = None
        self.vetor_x_tabuleiro = None
        self.vetor_y_tabuleiro = None
        self.matriz_transformacao = None
        
        print("🎯 Sistema de Matriz do Tabuleiro inicializado!")
        print(f"📏 Área: {largura_tabuleiro_mm}x{altura_tabuleiro_mm}x{altura_trabalho_mm}mm")
        print(f"🔢 Matriz: {self.pontos_x}x{self.pontos_y}x{self.pontos_z} pontos")
        print(f"📍 Marcadores de referência: {self.id_marcador_ref1}, {self.id_marcador_ref2}")
    
    def detectar_marcadores(self, frame: np.ndarray) -> bool:
        """Detecta todos os marcadores no frame."""
        # Limpar detecções anteriores
        self.marcadores_referencia.clear()
        self.marcadores_pecas.clear()
        
        # Detectar marcadores ArUco
        corners, ids, _ = cv2.aruco.detectMarkers(
            frame, self.aruco_dict, parameters=self.parameters
        )
        
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Processar cada marcador detectado
            for i, marker_id in enumerate(ids):
                marker_id = marker_id[0]
                
                # Estimar pose do marcador
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners[i:i+1], self.tamanho_marcador_metros,
                    self.camera_matrix, self.dist_coeffs
                )
                
                info_marcador = {
                    'id': marker_id,
                    'posicao_mundo': tvecs[0][0],
                    'rotacao': rvecs[0][0],
                    'corners': corners[i],
                    'timestamp': time.time(),
                    'confianca': self._calcular_confianca(corners[i])
                }
                
                # Classificar marcador
                if marker_id in [self.id_marcador_ref1, self.id_marcador_ref2]:
                    self.marcadores_referencia[marker_id] = info_marcador
                    self._desenhar_marcador_referencia(frame, info_marcador)
                else:
                    self.marcadores_pecas[marker_id] = info_marcador
                    self._desenhar_marcador_peca(frame, info_marcador)
            
            # Calibrar sistema se possível
            if len(self.marcadores_referencia) == 2:
                self._calibrar_sistema_tabuleiro()
            
            return True
        
        return False
    
    def _calibrar_sistema_tabuleiro(self):
        """Calibra o sistema de coordenadas baseado nos marcadores de referência."""
        if (self.id_marcador_ref1 not in self.marcadores_referencia or 
            self.id_marcador_ref2 not in self.marcadores_referencia):
            return False
        
        # Posições dos marcadores de referência
        pos_ref1 = self.marcadores_referencia[self.id_marcador_ref1]['posicao_mundo']
        pos_ref2 = self.marcadores_referencia[self.id_marcador_ref2]['posicao_mundo']
        
        # Definir origem no marcador 1
        self.origem_tabuleiro = pos_ref1.copy()
        
        # Calcular vetor X (direção do marcador 1 para marcador 2)
        vetor_ref = pos_ref2 - pos_ref1
        self.vetor_x_tabuleiro = vetor_ref / np.linalg.norm(vetor_ref)
        
        # Calcular vetor Y (perpendicular ao X no plano do tabuleiro)
        # Assumindo que o tabuleiro está no plano XY
        vetor_z_mundo = np.array([0, 0, 1])  # Eixo Z para cima
        self.vetor_y_tabuleiro = np.cross(vetor_z_mundo, self.vetor_x_tabuleiro)
        self.vetor_y_tabuleiro = self.vetor_y_tabuleiro / np.linalg.norm(self.vetor_y_tabuleiro)
        
        # Criar matriz de transformação mundo -> tabuleiro
        self.matriz_transformacao = np.column_stack([
            self.vetor_x_tabuleiro,
            self.vetor_y_tabuleiro,
            np.cross(self.vetor_x_tabuleiro, self.vetor_y_tabuleiro)
        ])
        
        # Calcular escala baseada na distância entre marcadores
        distancia_real_marcadores = np.linalg.norm(vetor_ref) * 1000  # converter para mm
        self.escala_x = self.largura_tabuleiro_mm / distancia_real_marcadores
        
        self.sistema_calibrado = True
        print(f"✅ Sistema calibrado! Escala X: {self.escala_x:.3f}")
        
        return True
    
    def converter_para_coordenadas_tabuleiro(self, posicao_mundo: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """
        Converte posição do mundo para coordenadas da matriz do tabuleiro.
        
        Args:
            posicao_mundo: Posição em coordenadas do mundo (metros)
            
        Returns:
            Tupla (x, y, z) na matriz do tabuleiro ou None se fora dos limites
        """
        if not self.sistema_calibrado:
            return None
        
        # Transladar para origem do tabuleiro
        pos_relativa = posicao_mundo - self.origem_tabuleiro
        
        # Projetar nos eixos do tabuleiro
        x_tabuleiro = np.dot(pos_relativa, self.vetor_x_tabuleiro) * 1000 * self.escala_x  # em mm
        y_tabuleiro = np.dot(pos_relativa, self.vetor_y_tabuleiro) * 1000 * self.escala_x  # em mm
        z_tabuleiro = self.altura_trabalho_mm  # Altura fixa de trabalho
        
        # Converter para índices da matriz
        indice_x = int(x_tabuleiro / self.resolucao_mm)
        indice_y = int(y_tabuleiro / self.resolucao_mm)
        indice_z = int(z_tabuleiro / self.resolucao_mm)
        
        # Verificar limites
        if (0 <= indice_x < self.pontos_x and 
            0 <= indice_y < self.pontos_y and 
            0 <= indice_z < self.pontos_z):
            return (indice_x, indice_y, indice_z)
        
        return None
    
    def converter_indice_para_coordenadas_mundo(self, indice_x: int, indice_y: int, indice_z: int) -> Optional[np.ndarray]:
        """
        Converte índices da matriz para coordenadas do mundo.
        
        Args:
            indice_x, indice_y, indice_z: Índices na matriz
            
        Returns:
            Coordenadas do mundo em metros ou None se inválido
        """
        if (not self.sistema_calibrado or 
            not (0 <= indice_x < self.pontos_x) or
            not (0 <= indice_y < self.pontos_y) or
            not (0 <= indice_z < self.pontos_z)):
            return None
        
        # Converter índices para mm
        x_mm = indice_x * self.resolucao_mm
        y_mm = indice_y * self.resolucao_mm
        z_mm = indice_z * self.resolucao_mm
        
        # Converter para metros no sistema do tabuleiro
        x_tabuleiro = x_mm / 1000.0 / self.escala_x
        y_tabuleiro = y_mm / 1000.0 / self.escala_x
        z_tabuleiro = z_mm / 1000.0
        
        # Converter para coordenadas do mundo
        pos_mundo = (self.origem_tabuleiro + 
                    x_tabuleiro * self.vetor_x_tabuleiro + 
                    y_tabuleiro * self.vetor_y_tabuleiro + 
                    z_tabuleiro * np.array([0, 0, 1]))
        
        return pos_mundo
    
    def obter_posicoes_pecas(self) -> Dict[int, Dict[str, Any]]:
        """Retorna posições de todas as peças na matriz do tabuleiro."""
        posicoes_pecas = {}
        
        for marker_id, info in self.marcadores_pecas.items():
            posicao_mundo = info['posicao_mundo']
            coordenadas_matriz = self.converter_para_coordenadas_tabuleiro(posicao_mundo)
            
            if coordenadas_matriz:
                # Calcular coordenadas em mm
                x_mm = coordenadas_matriz[0] * self.resolucao_mm
                y_mm = coordenadas_matriz[1] * self.resolucao_mm
                z_mm = self.altura_trabalho_mm
                
                posicoes_pecas[marker_id] = {
                    'id': marker_id,
                    'indices_matriz': coordenadas_matriz,
                    'coordenadas_mm': (x_mm, y_mm, z_mm),
                    'coordenadas_mundo_metros': posicao_mundo,
                    'confianca': info['confianca'],
                    'timestamp': info['timestamp']
                }
        
        return posicoes_pecas
    
    def obter_ponto_matriz(self, x: int, y: int, z: int = None) -> Optional[Dict[str, Any]]:
        """
        Obtém informações de um ponto específico da matriz.
        
        Args:
            x, y: Coordenadas na matriz
            z: Coordenada Z (opcional, usa altura de trabalho por padrão)
            
        Returns:
            Informações do ponto ou None se inválido
        """
        if z is None:
            z = self.pontos_z // 2  # Meio da altura de trabalho
        
        coordenadas_mundo = self.converter_indice_para_coordenadas_mundo(x, y, z)
        if coordenadas_mundo is None:
            return None
        
        return {
            'indices_matriz': (x, y, z),
            'coordenadas_mm': (x * self.resolucao_mm, y * self.resolucao_mm, z * self.resolucao_mm),
            'coordenadas_mundo_metros': coordenadas_mundo,
            'dentro_tabuleiro': True
        }
    
    def mostrar_informacoes_tela(self, frame: np.ndarray) -> None:
        """Mostra informações do sistema na tela."""
        altura_texto = 30
        
        # Status do sistema
        if self.sistema_calibrado:
            cv2.putText(frame, "SISTEMA CALIBRADO", (10, altura_texto), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "AGUARDANDO MARCADORES DE REFERENCIA", (10, altura_texto), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"Necessario: IDs {self.id_marcador_ref1} e {self.id_marcador_ref2}", (10, altura_texto + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
        
        altura_texto += 60
        
        # Informações dos marcadores de referência
        cv2.putText(frame, f"REFERENCIAS: {len(self.marcadores_referencia)}/2", (10, altura_texto), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        altura_texto += 25
        
        # Informações das peças
        cv2.putText(frame, f"PECAS: {len(self.marcadores_pecas)}", (10, altura_texto), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        altura_texto += 25
        
        # Mostrar posições das peças
        if self.sistema_calibrado and self.marcadores_pecas:
            posicoes = self.obter_posicoes_pecas()
            for marker_id, pos in posicoes.items():
                coords = pos['coordenadas_mm']
                texto = f"ID{marker_id}: ({coords[0]:.0f},{coords[1]:.0f})mm"
                cv2.putText(frame, texto, (10, altura_texto), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                altura_texto += 18
    
    def _desenhar_marcador_referencia(self, frame: np.ndarray, info: Dict[str, Any]) -> None:
        """Desenha marcador de referência com destaque."""
        corners = info['corners']
        centro = np.mean(corners[0], axis=0).astype(int)
        
        # Destacar com círculo vermelho
        cv2.circle(frame, tuple(centro), 30, (0, 0, 255), 3)
        
        # Texto do ID
        cv2.putText(frame, f"REF{info['id']}", (centro[0] + 35, centro[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    def _desenhar_marcador_peca(self, frame: np.ndarray, info: Dict[str, Any]) -> None:
        """Desenha marcador de peça."""
        corners = info['corners']
        centro = np.mean(corners[0], axis=0).astype(int)
        
        # Destacar com círculo verde
        cv2.circle(frame, tuple(centro), 20, (0, 255, 0), 2)
        
        # Texto do ID
        cv2.putText(frame, f"P{info['id']}", (centro[0] + 25, centro[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Mostrar coordenadas se calibrado
        if self.sistema_calibrado:
            coords = self.converter_para_coordenadas_tabuleiro(info['posicao_mundo'])
            if coords:
                coords_mm = (coords[0] * self.resolucao_mm, coords[1] * self.resolucao_mm)
                texto_coords = f"({coords_mm[0]:.0f},{coords_mm[1]:.0f})"
                cv2.putText(frame, texto_coords, (centro[0] + 25, centro[1] + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _calcular_confianca(self, corners: np.ndarray) -> float:
        """Calcula confiança baseada na qualidade dos cantos."""
        area = cv2.contourArea(corners[0])
        return min(1.0, area / 5000.0)
    
    def salvar_configuracao_sistema(self, filename: str = "sistema_tabuleiro.json") -> bool:
        """Salva configuração e dados do sistema."""
        try:
            dados = {
                'configuracao': {
                    'largura_tabuleiro_mm': self.largura_tabuleiro_mm,
                    'altura_tabuleiro_mm': self.altura_tabuleiro_mm,
                    'altura_trabalho_mm': self.altura_trabalho_mm,
                    'resolucao_mm': self.resolucao_mm,
                    'matriz_pontos': f"{self.pontos_x}x{self.pontos_y}x{self.pontos_z}",
                    'sistema_calibrado': self.sistema_calibrado
                },
                'posicoes_pecas': self.obter_posicoes_pecas(),
                'timestamp': time.time()
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(dados, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Configuração salva em: {filename}")
            return True
        except Exception as e:
            print(f"❌ Erro ao salvar: {e}")
            return False


# Classe para controle da garra robótica
class ControleGarraRobotica:
    """Interface para controle de garra robótica usando o sistema de matriz."""
    
    def __init__(self, sistema_tabuleiro: SistemaMatrizTabuleiro):
        self.sistema = sistema_tabuleiro
        self.posicao_atual = (0, 0, 0)
    
    def mover_para_peca(self, marker_id: int) -> bool:
        """Move a garra para uma peça específica."""
        posicoes = self.sistema.obter_posicoes_pecas()
        
        if marker_id not in posicoes:
            print(f"❌ Peça {marker_id} não encontrada")
            return False
        
        pos = posicoes[marker_id]
        coords_mm = pos['coordenadas_mm']
        
        print(f"🤖 Movendo garra para peça {marker_id}:")
        print(f"   Posição: ({coords_mm[0]:.1f}, {coords_mm[1]:.1f}, {coords_mm[2]:.1f}) mm")
        print(f"   Índices matriz: {pos['indices_matriz']}")
        print(f"   Confiança: {pos['confianca']*100:.1f}%")
        
        self.posicao_atual = coords_mm
        return True
    
    def mover_para_coordenada(self, x_mm: float, y_mm: float, z_mm: float = None) -> bool:
        """Move a garra para uma coordenada específica."""
        if z_mm is None:
            z_mm = self.sistema.altura_trabalho_mm
        
        # Converter mm para índices da matriz
        indice_x = int(x_mm / self.sistema.resolucao_mm)
        indice_y = int(y_mm / self.sistema.resolucao_mm)
        indice_z = int(z_mm / self.sistema.resolucao_mm)
        
        ponto = self.sistema.obter_ponto_matriz(indice_x, indice_y, indice_z)
        if ponto is None:
            print(f"❌ Coordenada ({x_mm}, {y_mm}, {z_mm}) fora dos limites")
            return False
        
        print(f"🤖 Movendo garra para coordenada:")
        print(f"   Posição: ({x_mm:.1f}, {y_mm:.1f}, {z_mm:.1f}) mm")
        print(f"   Índices matriz: {ponto['indices_matriz']}")
        
        self.posicao_atual = (x_mm, y_mm, z_mm)
        return True


def main():
    """Função principal do sistema de matriz do tabuleiro."""
    print("🎯 === SISTEMA DE MATRIZ DO TABULEIRO ===")
    print("📋 CONFIGURAÇÃO INICIAL:")
    print("   - Coloque marcador ID 0 no canto inferior esquerdo")
    print("   - Coloque marcador ID 1 no canto inferior direito") 
    print("   - Coloque peças com outros IDs (2, 3, 4...) no tabuleiro")
    
    try:
        # Inicializar sistema (ajustar dimensões conforme seu tabuleiro)
        sistema = SistemaMatrizTabuleiro(
            largura_tabuleiro_mm=300,  # 30cm de largura
            altura_tabuleiro_mm=200,   # 20cm de altura
            altura_trabalho_mm=50      # 5cm de altura de trabalho
        )
        
        # Inicializar controle da garra
        garra = ControleGarraRobotica(sistema)
        
        # Tentar abrir câmera
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("⚠️  Câmera não disponível.")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n🎮 CONTROLES:")
        print("   ESC = Sair")
        print("   S = Salvar configuração")
        print("   2-9 = Mover garra para peça específica")
        print("   C = Mover para centro do tabuleiro")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detectar todos os marcadores
            detectou = sistema.detectar_marcadores(frame)
            
            # Mostrar informações
            sistema.mostrar_informacoes_tela(frame)
            
            # Desenhar grade do tabuleiro (se calibrado)
            if sistema.sistema_calibrado:
                cv2.putText(frame, "SISTEMA CALIBRADO - PRONTO PARA USO", 
                           (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("🎯 Sistema Matriz Tabuleiro", frame)
            
            # Processar teclas
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('s') or key == ord('S'):
                sistema.salvar_configuracao_sistema()
            elif key == ord('c') or key == ord('C'):
                # Mover para centro do tabuleiro
                centro_x = sistema.largura_tabuleiro_mm // 2
                centro_y = sistema.altura_tabuleiro_mm // 2
                garra.mover_para_coordenada(centro_x, centro_y)
            elif key in [ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9')]:
                # Mover garra para peça específica
                peca_id = key - ord('0')
                garra.mover_para_peca(peca_id)
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("🎯 Sistema finalizado!")
        
    except KeyboardInterrupt:
        print("\n🛑 Sistema interrompido")
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()