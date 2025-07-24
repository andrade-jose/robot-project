from enum import Enum

class FaseJogo(Enum):
    COLOCACAO = "colocacao"
    MOVIMENTO = "movimento" 
    JOGO_TERMINADO = "jogo_terminado"

class Jogador(Enum):
    JOGADOR1 = 1  # Peças do robô
    JOGADOR2 = 2  # Peças do humano
    VAZIO = 0


CALIB: str= r'C:\Venv\Rep_git\robot-project\robotics_project\data\stereo_dataset\calib.plk'