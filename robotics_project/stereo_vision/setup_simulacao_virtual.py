# setup_simulacao_virtual.py
import numpy as np

def ativar_simulacao_virtual(tapatan_robotico):
    """
    Injeta coordenadas mock (virtuais) no robô Tapatan e simula a câmera estéreo.
    Chame isso no início do jogo para testar sem câmera.
    """
    coords_mock = np.array([
        [0.3, 0.2, 0.01], [0.4, 0.2, 0.01], [0.5, 0.2, 0.01],
        [0.3, 0.1, 0.01], [0.4, 0.1, 0.01], [0.5, 0.1, 0.01],
        [0.3, 0.0, 0.01], [0.4, 0.0, 0.01], [0.5, 0.0, 0.01],
    ])
    tapatan_robotico.definir_coordenadas_tabuleiro(coords_mock)
    print("✅ Coordenadas simuladas aplicadas (modo virtual ativo)")
