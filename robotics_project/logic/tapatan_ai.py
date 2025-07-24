from typing import List, Tuple, Optional
import sys
from pathlib import Path

# Add project root to Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from config.config import Jogador
from logic.tapatan_logic import TabuleiraTapatan

class TapatanAI:

    def __init__(self, jogo: TabuleiraTapatan):
        self.jogo = jogo

    def avaliar_tabuleiro(self, tabuleiro: List[Jogador]) -> int:
        """Função de avaliação: +10 vitória do robô, -10 vitória do humano, 0 se neutro"""
        vencedor = self._verificar_vencedor_tabuleiro(tabuleiro)
        if vencedor == Jogador.JOGADOR1:
            return 10
        elif vencedor == Jogador.JOGADOR2:
            return -10
        return 0

    def minimax(self, tabuleiro: List[Jogador], profundidade: int, maximizando: bool) -> int:
        score = self.avaliar_tabuleiro(tabuleiro)

        if abs(score) == 10 or profundidade == 0:
            return score

        jogador = Jogador.JOGADOR1 if maximizando else Jogador.JOGADOR2
        pecas = [i for i, peca in enumerate(tabuleiro) if peca == jogador]

        if maximizando:
            melhor_valor = -float('inf')
            for origem in pecas:
                for destino in self.jogo.mapa_adjacencia[origem]:
                    if tabuleiro[destino] == Jogador.VAZIO:
                        novo_tabuleiro = tabuleiro.copy()
                        novo_tabuleiro[origem] = Jogador.VAZIO
                        novo_tabuleiro[destino] = Jogador.JOGADOR1
                        valor = self.minimax(novo_tabuleiro, profundidade - 1, False)
                        melhor_valor = max(melhor_valor, valor)
            return melhor_valor
        else:
            pior_valor = float('inf')
            for origem in pecas:
                for destino in self.jogo.mapa_adjacencia[origem]:
                    if tabuleiro[destino] == Jogador.VAZIO:
                        novo_tabuleiro = tabuleiro.copy()
                        novo_tabuleiro[origem] = Jogador.VAZIO
                        novo_tabuleiro[destino] = Jogador.JOGADOR2
                        valor = self.minimax(novo_tabuleiro, profundidade - 1, True)
                        pior_valor = min(pior_valor, valor)
            return pior_valor

    def fazer_jogada_robo_minimax(self, profundidade: int = 3) -> Optional[Tuple[int, int]]:
        """IA do robô com Minimax: retorna (origem, destino)"""
        melhor_valor = -float('inf')
        melhor_jogada = None
        tabuleiro = self.jogo.tabuleiro.copy()
        pecas = self.jogo.obter_pecas_jogador(Jogador.JOGADOR1)

        for origem in pecas:
            for destino in self.jogo.mapa_adjacencia[origem]:
                if tabuleiro[destino] == Jogador.VAZIO:
                    novo_tabuleiro = tabuleiro.copy()
                    novo_tabuleiro[origem] = Jogador.VAZIO
                    novo_tabuleiro[destino] = Jogador.JOGADOR1
                    valor = self.minimax(novo_tabuleiro, profundidade - 1, False)

                    if valor > melhor_valor:
                        melhor_valor = valor
                        melhor_jogada = (origem, destino)

        return melhor_jogada