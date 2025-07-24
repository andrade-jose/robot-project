"""
Lógica do Jogo Tapatan para Implementação em Braço Robótico
Jogo tradicional filipino de estratégia 3x3

FORMATO DO TABULEIRO TAPATAN:
O tabuleiro é um quadrado com linhas bissetrizes formando 9 pontos de intersecção:

    0-----1-----2
    | \   |   / |
    |   \ | /   |
    3-----4-----5
    |   / | \   |
    | /   |   \ |
    6-----7-----8

Regras do Jogo:
- Tabuleiro com grade 3x3 e 9 posições numeradas de 0 a 8
- Cada jogador tem exatamente 3 peças
- Duas fases: Colocação, depois Movimento  

- Vitória ao conseguir 3 peças em linha (horizontal, vertical ou diagonal)
- Movimento apenas para posições adjacentes vazias seguindo as linhas do tabuleiro
"""

from typing import List, Tuple, Optional
from config.config import FaseJogo, Jogador



class TabuleiraTapatan:
    """
    Representação do tabuleiro:
    Posições numeradas de 0 a 8:
    0 1 2
    3 4 5  
    6 7 8
    
    Cada posição pode conter: VAZIO, JOGADOR1, ou JOGADOR2
    """
    
    def __init__(self):
        # Tabuleiro 3x3 representado como array 1D para facilitar indexação
        self.tabuleiro = [Jogador.VAZIO] * 9
        self.tabuleiro[0] = Jogador.JOGADOR1
        self.tabuleiro[2] = Jogador.JOGADOR1
        self.tabuleiro[7] = Jogador.JOGADOR1

        self.tabuleiro[1] = Jogador.JOGADOR2
        self.tabuleiro[6] = Jogador.JOGADOR2
        self.tabuleiro[8] = Jogador.JOGADOR2

        self.pecas_colocadas = {
            Jogador.JOGADOR1: 3,
            Jogador.JOGADOR2: 3
        }
        self.fase = FaseJogo.MOVIMENTO
        self.jogador_atual = Jogador.JOGADOR1

        # Define as combinações vencedoras (3 em linha)
        self.padroes_vitoria = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Linhas horizontais
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Linhas verticais  
            [0, 4, 8], [2, 4, 6]              # Linhas diagonais
        ]
        
        # Define posições adjacentes para a fase de movimento
        # Baseado nas linhas do tabuleiro Tapatan tradicional
        self.mapa_adjacencia = {
            0: [1, 3, 4],     
            1: [0, 2, 4],     
            2: [1, 4, 5],          
            3: [0, 4, 6],     
            4: [0, 1, 2, 3, 5, 6, 7, 8],  # Centro (conecta a todos)
            5: [2, 4, 8],    
            6: [3, 4, 7],           
            7: [4, 6, 8],     
            8: [4, 5, 7]            
        }

    def obter_estado_tabuleiro(self) -> List[int]:
        """Retorna o estado atual do tabuleiro como lista de inteiros"""
        return [jogador.value for jogador in self.tabuleiro]

    def obter_posicoes_vazias(self) -> List[int]:
        """Obtém todas as posições vazias no tabuleiro"""
        return [i for i, peca in enumerate(self.tabuleiro) if peca == Jogador.VAZIO]

    def eh_movimento_valido(self, pos_origem: int, pos_destino: int) -> bool:
        """Verifica se o movimento é válido durante a fase de movimento"""
        if self.fase != FaseJogo.MOVIMENTO:
            return False
        
        # Verifica se pos_origem tem peça do jogador atual
        if self.tabuleiro[pos_origem] != self.jogador_atual:
            return False
            
        # Verifica se pos_destino está vazia
        if self.tabuleiro[pos_destino] != Jogador.VAZIO:
            return False
            
        # Verifica se as posições são adjacentes (conectadas por linha)
        return pos_destino in self.mapa_adjacencia[pos_origem]


    def fazer_movimento(self, pos_origem: int, pos_destino: int) -> bool:
        """Move uma peça durante a fase de movimento"""
        if not self.eh_movimento_valido(pos_origem, pos_destino):
            return False
            
        # Move a peça: remove da origem, coloca no destino
        self.tabuleiro[pos_origem] = Jogador.VAZIO
        self.tabuleiro[pos_destino] = self.jogador_atual
        return True

    def verificar_vencedor(self) -> Optional[Jogador]:
        """Verifica se há um vencedor"""
        for padrao in self.padroes_vitoria:
            if (self.tabuleiro[padrao[0]] == self.tabuleiro[padrao[1]] == 
                self.tabuleiro[padrao[2]] != Jogador.VAZIO):
                return self.tabuleiro[padrao[0]]
        return None

    def obter_pecas_jogador(self, jogador: Jogador) -> List[int]:
        """Obtém posições de todas as peças de um jogador específico"""
        return [i for i, peca in enumerate(self.tabuleiro) if peca == jogador]

    def obter_movimentos_validos(self, jogador: Jogador = None) -> List[Tuple[int, int]]:
        if jogador is None:
            jogador = self.jogador_atual

        if self.fase != FaseJogo.MOVIMENTO:
            return []

        movimentos_validos = []
        pecas_jogador = self.obter_pecas_jogador(jogador)

        for pos_origem in pecas_jogador:
            for pos_destino in self.mapa_adjacencia[pos_origem]:  # ✅ usa conexões diretas
                if self.tabuleiro[pos_destino] == Jogador.VAZIO:
                    movimentos_validos.append((pos_origem, pos_destino))

        return movimentos_validos

    def alternar_jogador(self):
        """Alterna para o outro jogador"""
        self.jogador_atual = Jogador.JOGADOR1 if self.jogador_atual == Jogador.JOGADOR2 else Jogador.JOGADOR2

    def jogo_terminado(self) -> bool:
        """Verifica se o jogo terminou"""
        vencedor = self.verificar_vencedor()
        if vencedor:
            self.fase = FaseJogo.JOGO_TERMINADO
            return True
        
        # Verifica empate por ausência de movimentos válidos
        if self.fase == FaseJogo.MOVIMENTO and not self.obter_movimentos_validos():
            self.fase = FaseJogo.JOGO_TERMINADO
            return True
            
        return False

    def posicao_para_coordenadas(self, posicao: int) -> Tuple[int, int]:
        """Converte posição do tabuleiro para coordenadas (linha, coluna)"""
        return (posicao // 3, posicao % 3)

    def coordenadas_para_posicao(self, linha: int, coluna: int) -> int:
        """Converte coordenadas (linha, coluna) para posição do tabuleiro"""
        return linha * 3 + coluna

    def imprimir_tabuleiro(self):
        """Imprime o estado atual do tabuleiro para depuração"""
        simbolos = {Jogador.VAZIO: '.', Jogador.JOGADOR1: 'X', Jogador.JOGADOR2: 'O'}
        print(f"Fase: {self.fase.value}, Jogador Atual: {self.jogador_atual.name}")
        print("Tabuleiro:")
        for i in range(3):
            linha = [simbolos[self.tabuleiro[i*3 + j]] for j in range(3)]
            print(' | '.join(linha))
            if i < 2:
                print('---------')
        print()