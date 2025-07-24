import numpy as np
from typing import List, Tuple, Optional, Dict
import sys
from pathlib import Path

# Add project root to Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from logic.tapatan_logic import TabuleiraTapatan
from config.config import FaseJogo, Jogador
from logic.tapatan_ai import TapatanAI
from stereo_vision.coordinate_transformer import CoordinateTransformer
from control.ur_controller import URController

class TapatanRobotico:
    def __init__(self):
        self.jogo = TabuleiraTapatan()
        self.ia = TapatanAI(self.jogo)

        self.coordenadas_tabuleiro = {}  # pos_index: [x, y, z]
        self.altura_pegar_peca = 0.05
        self.altura_colocar_peca = 0.02

        self.transformador = CoordinateTransformer()
        self.transformador.load_transformation()

        self.ur = URController()

    def definir_coordenadas_tabuleiro(self, matriz_coordenadas: np.ndarray):
        if matriz_coordenadas.shape != (9, 3):
            raise ValueError("Matriz de coordenadas deve ser 9x3.")
        for i in range(9):
            self.coordenadas_tabuleiro[i] = matriz_coordenadas[i]

    def obter_sequencia_movimento(self, pos_origem: int, pos_destino: int) -> List[List[float]]:
        coord_origem = self.coordenadas_tabuleiro[pos_origem]
        coord_destino = self.coordenadas_tabuleiro[pos_destino]

        return [
            [*coord_origem, 0, 0, 0],  # acima da peça
            [coord_origem[0], coord_origem[1], coord_origem[2] + self.altura_colocar_peca, 0, 0, 0],  # pegar
            [*coord_origem, 0, 0, 0],
            [*coord_destino, 0, 0, 0],  # ir para destino
            [coord_destino[0], coord_destino[1], coord_destino[2] + self.altura_colocar_peca, 0, 0, 0],  # soltar
            [*coord_destino, 0, 0, 0]  # subir
        ]

    def processar_entrada_visao(self, saida_classificador: List[int]) -> Dict:
        if len(saida_classificador) != 9:
            raise ValueError("Saída do classificador deve ter 9 valores.")
        for i, val in enumerate(saida_classificador):
            self.jogo.tabuleiro[i] = Jogador(val)

        resultado = {
            'estado_tabuleiro': self.jogo.obter_estado_tabuleiro(),
            'fase': self.jogo.fase.value,
            'jogador_atual': self.jogo.jogador_atual.name,
            'movimentos_validos': self.jogo.obter_movimentos_validos(),
            'vencedor': None,
            'jogo_terminado': self.jogo.jogo_terminado()
        }

        vencedor = self.jogo.verificar_vencedor()
        if vencedor:
            resultado['vencedor'] = vencedor.name

        return resultado

    def fazer_jogada_robo(self) -> Optional[Tuple[int, int]]:
        if self.jogo.fase == FaseJogo.COLOCACAO:
            movimentos = self.jogo.obter_movimentos_validos(Jogador.JOGADOR1)
            for pos in [4, 0, 2, 6, 8, 1, 3, 5, 7]:
                if (pos, -1) in movimentos:
                    return (pos, -1)
            return movimentos[0] if movimentos else None

        elif self.jogo.fase == FaseJogo.MOVIMENTO:
            jogada = self.ia.fazer_jogada_robo_minimax(profundidade=3)
            if jogada:
                self.executar_movimento_robo(jogada[0], jogada[1])
            return jogada
        return None

    def executar_movimento_robo(self, origem: int, destino: int):
        """Executa o movimento físico com o braço robótico"""
        sequencia = self.obter_sequencia_movimento(origem, destino)
        for pose in sequencia:
            self.ur.move_to_pose(pose)