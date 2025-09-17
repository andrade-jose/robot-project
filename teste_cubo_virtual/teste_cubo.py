#!/usr/bin/env python3
"""
Sistema de Cubo Virtual - Python
=================================
Detecta marcadores ArUco e sobrepõe cubo virtual 3D
Fornece matriz de coordenadas e informações espaciais em tempo real
"""

import cv2
import numpy as np
import time
import json
import os
from typing import List, Tuple, Optional, Dict, Any

class CuboVirtual:
    """Sistema principal de detecção e rastreamento de cubo virtual."""
    
    def __init__(self):
        """Inicializa o sistema de cubo virtual."""
        # Configurar detector ArUco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.parameters = cv2.aruco.DetectorParameters()
        
        # Parâmetros da câmera (calibre para maior precisão)
        self.camera_matrix = np.array([
            [800, 0, 320],    # fx, 0, cx
            [0, 800, 240],    # 0, fy, cy
            [0, 0, 1]         # 0, 0, 1
        ], dtype=np.float64)
        
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        
        # Informações do objeto atual
        self.objeto_atual = {
            'posicao': np.array([0.0, 0.0, 0.0]),
            'rotacao': np.array([0.0, 0.0, 0.0]),
            'matriz_4x4': np.eye(4),
            'confianca': 0.0,
            'visivel': False,
            'timestamp': 0.0,
            'id_marcador': -1
        }
        
        # Histórico de detecções
        self.historico_deteccoes: List[Dict[str, Any]] = []
        
        print("🎯 Sistema de Cubo Virtual Python inicializado!")
        print("📋 Para testar: imprima um marcador ArUco ID=0")
        print("🔗 Link: https://chev.me/arucogen/ (selecione 6x6, ID=0)")
    
    def detectar_marcador(self, frame: np.ndarray, marker_id: int = 0) -> bool:
        """
        Detecta marcador ArUco e calcula posição do cubo virtual.
        
        Args:
            frame: Imagem da câmera
            marker_id: ID do marcador a detectar
            
        Returns:
            True se detectou o marcador, False caso contrário
        """
        # Detectar marcadores ArUco
        corners, ids, _ = cv2.aruco.detectMarkers(
            frame, self.aruco_dict, parameters=self.parameters
        )
        
        if ids is not None:
            # Desenhar todos os marcadores detectados
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Procurar pelo marcador específico
            for i, detected_id in enumerate(ids):
                if detected_id[0] == marker_id:
                    # Estimar pose do marcador (assumindo 5cm de lado)
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners[i:i+1], 0.05,  # 5cm de lado
                        self.camera_matrix, self.dist_coeffs
                    )
                    
                    # Atualizar informações do objeto
                    self.objeto_atual.update({
                        'posicao': tvecs[0][0],
                        'rotacao': rvecs[0][0],
                        'visivel': True,
                        'timestamp': time.time(),
                        'confianca': self._calcular_confianca(corners[i]),
                        'id_marcador': marker_id,
                        'matriz_4x4': self._calcular_matriz_4x4(rvecs[0][0], tvecs[0][0])
                    })
                    
                    # Salvar no histórico
                    self.historico_deteccoes.append(self.objeto_atual.copy())
                    if len(self.historico_deteccoes) > 100:
                        self.historico_deteccoes.pop(0)
                    
                    # Desenhar eixos 3D (X=vermelho, Y=verde, Z=azul)
                    self._desenhar_eixos_3d(frame, rvecs[0][0], tvecs[0][0])
                    
                    # Desenhar cubo virtual
                    self._desenhar_cubo_virtual(frame, rvecs[0][0], tvecs[0][0])
                    
                    return True
        
        self.objeto_atual['visivel'] = False
        return False
    
    def mostrar_informacoes(self, frame: np.ndarray) -> None:
        """Mostra informações do cubo na tela."""
        if self.objeto_atual['visivel']:
            pos = self.objeto_atual['posicao']
            rot = self.objeto_atual['rotacao']
            
            # Posição em centímetros
            pos_text = f"Pos(cm): {pos[0]*100:.1f}, {pos[1]*100:.1f}, {pos[2]*100:.1f}"
            
            # Rotação em graus
            rot_deg = rot * 180 / np.pi
            rot_text = f"Rot(deg): {rot_deg[0]:.1f}, {rot_deg[1]:.1f}, {rot_deg[2]:.1f}"
            
            # Confiança
            conf_text = f"Conf: {self.objeto_atual['confianca']*100:.0f}%"
            
            # Desenhar informações
            cv2.putText(frame, pos_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, rot_text, (10, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            cv2.putText(frame, conf_text, (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, "CUBO DETECTADO", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "PROCURANDO MARCADOR...", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Imprima marcador ArUco ID=0", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
    
    def imprimir_detalhes(self) -> None:
        """Imprime informações detalhadas do cubo."""
        if self.objeto_atual['visivel']:
            pos = self.objeto_atual['posicao']
            rot = self.objeto_atual['rotacao']
            
            print("\n=== 🎯 DETECÇÃO DO CUBO VIRTUAL ===")
            print("📍 Posição (metros):")
            print(f"   X: {pos[0]:.6f}")
            print(f"   Y: {pos[1]:.6f}")
            print(f"   Z: {pos[2]:.6f}")
            
            print("🔄 Rotação (radianos):")
            print(f"   RX: {rot[0]:.6f}")
            print(f"   RY: {rot[1]:.6f}")
            print(f"   RZ: {rot[2]:.6f}")
            
            print(f"📊 Confiança: {self.objeto_atual['confianca']*100:.1f}%")
            print(f"⏰ Timestamp: {self.objeto_atual['timestamp']:.3f}")
            print(f"🔢 Matriz de Transformação 4x4:")
            print(self.objeto_atual['matriz_4x4'])
            print(f"📈 Histórico: {len(self.historico_deteccoes)} detecções")
        else:
            print("❌ Nenhum marcador detectado no momento.")
    
    def salvar_dados(self, filename: str = "cubo_virtual_dados.json") -> bool:
        """Salva dados do cubo em arquivo JSON."""
        try:
            dados = {
                'objeto_atual': {
                    'posicao': self.objeto_atual['posicao'].tolist(),
                    'rotacao': self.objeto_atual['rotacao'].tolist(),
                    'matriz_4x4': self.objeto_atual['matriz_4x4'].tolist(),
                    'confianca': float(self.objeto_atual['confianca']),
                    'visivel': bool(self.objeto_atual['visivel']),
                    'timestamp': float(self.objeto_atual['timestamp']),
                    'id_marcador': int(self.objeto_atual['id_marcador'])
                },
                'historico_count': len(self.historico_deteccoes),
                'timestamp_salvo': time.time()
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(dados, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Dados salvos em: {filename}")
            return True
        except Exception as e:
            print(f"❌ Erro ao salvar dados: {e}")
            return False
    
    def obter_coordenadas(self) -> Dict[str, Any]:
        """Retorna dicionário com todas as coordenadas do cubo."""
        return {
            'posicao_xyz_metros': self.objeto_atual['posicao'].tolist(),
            'posicao_xyz_centimetros': (self.objeto_atual['posicao'] * 100).tolist(),
            'rotacao_xyz_radianos': self.objeto_atual['rotacao'].tolist(),
            'rotacao_xyz_graus': (self.objeto_atual['rotacao'] * 180 / np.pi).tolist(),
            'matriz_transformacao_4x4': self.objeto_atual['matriz_4x4'].tolist(),
            'confianca_percentual': self.objeto_atual['confianca'] * 100,
            'visivel': self.objeto_atual['visivel'],
            'timestamp': self.objeto_atual['timestamp'],
            'id_marcador': self.objeto_atual['id_marcador']
        }
    
    def calcular_velocidade_media(self) -> float:
        """Calcula velocidade média baseada no histórico."""
        if len(self.historico_deteccoes) < 2:
            return 0.0
        
        distancia_total = 0.0
        for i in range(1, len(self.historico_deteccoes)):
            pos_ant = self.historico_deteccoes[i-1]['posicao']
            pos_atual = self.historico_deteccoes[i]['posicao']
            distancia = np.linalg.norm(pos_atual - pos_ant)
            distancia_total += distancia
        
        tempo_total = (self.historico_deteccoes[-1]['timestamp'] - 
                      self.historico_deteccoes[0]['timestamp'])
        
        return distancia_total / tempo_total if tempo_total > 0 else 0.0
    
    def _calcular_confianca(self, corners: np.ndarray) -> float:
        """Calcula confiança baseada na qualidade dos cantos."""
        area = cv2.contourArea(corners[0])
        return min(1.0, area / 5000.0)
    
    def _calcular_matriz_4x4(self, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """Calcula matriz de transformação 4x4."""
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        return T
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
    
    def _desenhar_cubo_virtual(self, frame: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> None:
        """Desenha cubo virtual 3D sobre o marcador."""
        # Definir vértices do cubo (5cm de lado)
        cube_pts = np.array([
            # Base inferior
            [0, 0, 0], [0.05, 0, 0], [0.05, 0.05, 0], [0, 0.05, 0],
            # Topo superior
            [0, 0, -0.05], [0.05, 0, -0.05], [0.05, 0.05, -0.05], [0, 0.05, -0.05]
        ], dtype=np.float32)
        
        # Projetar pontos 3D para 2D
        img_pts, _ = cv2.projectPoints(cube_pts, rvec, tvec, 
                                     self.camera_matrix, self.dist_coeffs)
        img_pts = img_pts.astype(int)
        
        # Desenhar base (azul)
        cv2.polylines(frame, [img_pts[:4]], True, (255, 0, 0), 3)
        
        # Desenhar topo (verde)
        cv2.polylines(frame, [img_pts[4:8]], True, (0, 255, 0), 3)
        
        # Desenhar conexões verticais (vermelho)
        for i in range(4):
            cv2.line(frame, tuple(img_pts[i][0]), tuple(img_pts[i+4][0]), (0, 0, 255), 2)
        
        # Desenhar ponto central
        centro = ((img_pts[0] + img_pts[6]) // 2).flatten()
        cv2.circle(frame, tuple(centro), 5, (255, 255, 255), -1)


def main():
    """Função principal do sistema."""
    print("🎯 === SISTEMA CUBO VIRTUAL PYTHON ===")
    print(f"📹 OpenCV Version: {cv2.__version__}")
    
    try:
        # Inicializar sistema
        sistema = CuboVirtual()
        
        # Tentar abrir câmera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("⚠️  Câmera não disponível. Executando modo demonstração...")
            
            # Modo demonstração
            demo_img = np.zeros((500, 700, 3), dtype=np.uint8)
            cv2.putText(demo_img, "Sistema Cubo Virtual Python!", 
                       (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(demo_img, "Para testar:", (50, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(demo_img, "1. Conecte uma camera", (70, 230), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(demo_img, "2. Imprima marcador ArUco ID=0", (70, 260), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(demo_img, "3. Execute: python cubo_virtual.py", (70, 290), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.putText(demo_img, "Pressione qualquer tecla para sair", (50, 350), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Cubo Virtual - Modo Demonstracao", demo_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return
        
        # Configurar câmera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"📹 Câmera iniciada! Resolução: {width}x{height}")
        
        print("\n🎮 CONTROLES:")
        print("   ESC = Sair")
        print("   I = Mostrar informações detalhadas")
        print("   ESPAÇO = Pausar/Continuar")
        print("   S = Salvar dados")
        print("   C = Mostrar coordenadas")
        print("   V = Mostrar velocidade média")
        
        pausado = False
        frame_count = 0
        fps_timer = time.time()
        fps_atual = 0.0
        
        while True:
            if not pausado:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
            
            # Calcular FPS
            if frame_count % 30 == 0 and not pausado:
                current_time = time.time()
                fps_atual = 30 / (current_time - fps_timer)
                fps_timer = current_time
            
            # Detectar marcador e cubo virtual
            detectado = sistema.detectar_marcador(frame, marker_id=0)
            
            # Mostrar informações na tela
            sistema.mostrar_informacoes(frame)
            
            # Status do sistema
            status = "PAUSADO" if pausado else "ATIVO"
            cor_status = (0, 255, 255) if pausado else (0, 255, 0)
            cv2.putText(frame, status, (width - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_status, 2)
            
            # FPS e frame count
            cv2.putText(frame, f"FPS: {fps_atual:.1f}", (width - 120, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Frame: {frame_count}", (width - 120, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Instruções
            cv2.putText(frame, "ESC=Sair, I=Info, SPACE=Pause, S=Save, C=Coord", 
                       (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            cv2.imshow("🎯 Sistema Cubo Virtual Python", frame)
            
            # Processar teclas
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('i') or key == ord('I'):
                sistema.imprimir_detalhes()
            elif key == ord(' '):  # ESPAÇO
                pausado = not pausado
                print(f"Sistema {'PAUSADO' if pausado else 'RETOMADO'}")
            elif key == ord('s') or key == ord('S'):
                sistema.salvar_dados()
                cv2.imwrite("screenshot_cubo_virtual.jpg", frame)
                print("📸 Screenshot salvo: screenshot_cubo_virtual.jpg")
            elif key == ord('c') or key == ord('C'):
                coords = sistema.obter_coordenadas()
                print("\n📊 COORDENADAS ATUAIS:")
                for chave, valor in coords.items():
                    print(f"   {chave}: {valor}")
            elif key == ord('v') or key == ord('V'):
                velocidade = sistema.calcular_velocidade_media()
                print(f"\n🚀 Velocidade média: {velocidade:.4f} m/s")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n🎯 Sistema finalizado!")
        print(f"📊 Total de frames processados: {frame_count}")
        print(f"📈 Total de detecções: {len(sistema.historico_deteccoes)}")
        print(f"✅ Obrigado por usar o Sistema Cubo Virtual Python!")
        
    except KeyboardInterrupt:
        print("\n🛑 Sistema interrompido pelo usuário")
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()