#!/usr/bin/env python3
"""
Simulador Principal do Sistema Tapatan Robótico
Executa uma partida completa entre humano e IA com integração visão-robô
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
from typing import List

# Ajusta o caminho do projeto
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

from control.tapatan_robotic import TapatanRobotico
from config.config import FaseJogo, Jogador
from stereo_vision.coordinate_transformer import CoordinateTransformer


class TapatanTestSimulator:
    """Classe principal do simulador do sistema Tapatan robótico"""

    def __init__(self):
        self.tapatan = TapatanRobotico()
        self.coordinate_transformer = CoordinateTransformer()
        self.configurar_tabuleiro_com_transformacao()

    def configurar_tabuleiro_com_transformacao(self):
        """Define e transforma as posições do tabuleiro (câmera → robô)"""
        camera_coords = np.array([
            [-0.05,  0.05, 0.8], [0.0,  0.05, 0.8], [0.05,  0.05, 0.8],
            [-0.05,  0.0,  0.8], [0.0,  0.0,  0.8], [0.05,  0.0,  0.8],
            [-0.05, -0.05, 0.8], [0.0, -0.05, 0.8], [0.05, -0.05, 0.8]
        ])

        robot_coords = np.array([
            [0.3, 0.2, 0.01], [0.4, 0.2, 0.01], [0.5, 0.2, 0.01],
            [0.3, 0.1, 0.01], [0.4, 0.1, 0.01], [0.5, 0.1, 0.01],
            [0.3, 0.0, 0.01], [0.4, 0.0, 0.01], [0.5, 0.0, 0.01]
        ])

        try:
            self.coordinate_transformer.load_transformation()
            if self.coordinate_transformer.T is None or self.coordinate_transformer.T.size == 0:
                raise ValueError("Transformação carregada está vazia.")
            print("✅ Transformação carregada de arquivo")
        except (FileNotFoundError, ValueError):
            print("🔧 Calculando nova transformação...")
            self.coordinate_transformer.set_transformation_from_points(camera_coords, robot_coords)
            self.coordinate_transformer.save_transformation()
            print("✅ Transformação calculada e salva")

        coords_robo = np.array([
            self.coordinate_transformer.transform_point(p) for p in camera_coords
        ])

        self.tapatan.definir_coordenadas_tabuleiro(coords_robo)

        print("🎯 Matriz de transformação:\n", self.coordinate_transformer.T)
        print("\n📍 Coordenadas no sistema do robô:")
        for i, c in enumerate(coords_robo):
            print(f"  {i}: ({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})")

    def imprimir_tabuleiro(self, estado: List[int]):
        s = {0: '.', 1: 'X', 2: 'O'}
        print(f"\n{''.join([s[estado[i]] + ('-----' if i % 3 != 2 else '\n') for i in range(9)])}")

    def simular_deteccao_stereo_camera(self, estado: List[int]) -> tuple:
        coords_camera = {}
        for i, v in enumerate(estado):
            if v != 0:
                ruido = np.random.normal(0, 0.001, 3)
                coord = np.array([
                    -0.05 + (i % 3) * 0.05,
                     0.05 - (i // 3) * 0.05,
                     0.8
                ]) + ruido
                coords_camera[i] = coord
        return estado, coords_camera

    def transformar_deteccao_para_robo(self, estado: List[int]) -> tuple:
        classificacao, coords_camera = self.simular_deteccao_stereo_camera(estado)
        coords_robo = {
            i: self.coordinate_transformer.transform_point(c)
            for i, c in coords_camera.items()
        }
        resultado = self.tapatan.processar_entrada_visao(classificacao)
        return resultado, coords_robo

    def executar_movimento_robo(self, origem: int, destino: int):
        print(f"🤖 Movimento do robô de {origem} para {destino}")
        seq = self.tapatan.obter_sequencia_movimento(origem, destino)
        for i, (x, y, z) in enumerate(seq):
            print(f"  {i+1}. ({x:.3f}, {y:.3f}, {z:.3f})")

    def obter_jogada_humana(self, opcoes: List[tuple]):
        print("👤 Jogador humano: escolha uma jogada")
        for i, (o, d) in enumerate(opcoes):
            print(f"  {i+1}. {o} → {d}")
        while True:
            entrada = input("Escolha (número ou 'q' para sair): ")
            if entrada.lower() in ('q', 'quit', 'sair'):
                return None
            try:
                idx = int(entrada) - 1
                if 0 <= idx < len(opcoes):
                    return opcoes[idx]
            except ValueError:
                pass
            print("❌ Entrada inválida.")

    def executar_jogo(self):
        print("🎮 Iniciando jogo Tapatan robótico")

        estado = [1, 2, 1, 0, 0, 0, 2, 1, 2]
        resultado, _ = self.transformar_deteccao_para_robo(estado)
        self.imprimir_tabuleiro(resultado['estado_tabuleiro'])

        turno = 1
        while not resultado['jogo_terminado']:
            print(f"\n🔁 TURNO {turno}")
            jogador = resultado['jogador_atual']

            if jogador == 'JOGADOR1':
                jogada = self.tapatan.fazer_jogada_robo()
                if jogada:
                    o, d = jogada
                    self.executar_movimento_robo(o, d)
                    self.tapatan.jogo.fazer_movimento(o, d)
                else:
                    print("❌ IA não encontrou jogadas")
                    break
            else:
                opcoes = resultado['movimentos_validos']
                if not opcoes:
                    print("❌ Jogador humano sem jogadas")
                    break
                jogada = self.obter_jogada_humana(opcoes)
                if jogada is None:
                    print("🚪 Encerrado pelo jogador")
                    break
                o, d = jogada
                self.tapatan.jogo.fazer_movimento(o, d)

            novo_estado = self.tapatan.jogo.obter_estado_tabuleiro()
            resultado, _ = self.transformar_deteccao_para_robo(novo_estado)
            self.imprimir_tabuleiro(resultado['estado_tabuleiro'])

            self.tapatan.jogo.alternar_jogador()
            turno += 1

            if resultado['vencedor']:
                print(f"🏆 Vencedor: {resultado['vencedor']}")
                break
            if turno > 50:
                print("⚠️ Limite de turnos atingido")
                break

        print("🎮 Fim da partida")
        if resultado['vencedor']:
            v = "ROBÔ" if resultado['vencedor'] == 'JOGADOR1' else "HUMANO"
            print(f"🏆 {v} venceu!")
        else:
            print("🤝 Empate ou sem vencedor")


def main():
    try:
        print("🚀 Iniciando Simulador Tapatan")
        simulador = TapatanTestSimulator()
        simulador.executar_jogo()
    except KeyboardInterrupt:
        print("\n🛑 Encerrado pelo usuário.")
    except Exception as e:
        import traceback
        print(f"❌ Erro: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
