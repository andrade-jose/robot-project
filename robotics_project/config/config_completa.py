# ARQUIVO: config/config_completa.py
from dataclasses import dataclass
from typing import List, Tuple
import os
from enum import Enum

@dataclass
class ConfigRobo:
    # Conexão
    ip: str = "192.168.15.25"
    porta: int = 30004
    
    # Movimento
    velocidade_padrao: float = 0.1
    aceleracao_padrao: float = 0.1
    altura_segura: float = 0.1
    altura_pegar: float = 0.02
    
    # Timeouts
    timeout_movimento: float = 30.0
    timeout_conexao: float = 10.0
    
    # Workspace seguro
    workspace_x: Tuple[float, float] = (0.2, 0.8)
    workspace_y: Tuple[float, float] = (-0.4, 0.4)
    workspace_z: Tuple[float, float] = (0.0, 0.5)
    
    # Poses
    pose_home: List[float] = None
    
    def __post_init__(self):
        if self.pose_home is None:
            self.pose_home = [0.0, -0.3, 0.0, -0.3, 0.0, 0.0]


@dataclass  
class ConfigJogo:
    profundidade_ia: int = 3
    modo_debug: bool = False
    salvar_historico: bool = True
    
@dataclass
class ConfigSistema:
    # Arquivos
    arquivo_calibracao: str = 'data/stereo_dataset/calib.pkl'
    pasta_logs: str = 'logs'
    pasta_dados: str = 'data'
    
    # Visão
    usar_camera_real: bool = False
    fps_camera: int = 30
    
    def __post_init__(self):
        # Criar pastas se não existirem
        os.makedirs(self.pasta_logs, exist_ok=True)
        os.makedirs(self.pasta_dados, exist_ok=True)


class Jogador(Enum):
    VAZIO = 0
    JOGADOR1 = 1  # Robô/IA
    JOGADOR2 = 2  # Humano
    
    def __init__(self, value):
        self._value_ = value
    
    @classmethod
    def from_value(cls, value):
        """Método seguro para criar Jogador a partir de valor"""
        if isinstance(value, cls):
            return value
        if isinstance(value, int) and value in [0, 1, 2]:
            return cls(value)
        return cls.VAZIO

class FaseJogo(Enum):
    COLOCACAO = "colocacao"
    MOVIMENTO = "movimento"  
    JOGO_TERMINADO = "jogo_terminado"
    
    def __init__(self, value):
        self._value_ = value
    
    @classmethod
    def from_value(cls, value):
        """Método seguro para criar FaseJogo a partir de valor"""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            mapping = {
                "colocacao": cls.COLOCACAO,
                "movimento": cls.MOVIMENTO,
                "jogo_terminado": cls.JOGO_TERMINADO
            }
            return mapping.get(value.lower(), cls.COLOCACAO)
        return cls.COLOCACAO


# Instância global
CONFIG = {
    'robo': ConfigRobo(),
    'jogo': ConfigJogo(), 
    'sistema': ConfigSistema(),
}