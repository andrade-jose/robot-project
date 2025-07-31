# ARQUIVO: config/config_completa.py
from dataclasses import dataclass
from typing import List, Tuple
import os
from enum import Enum

@dataclass
class ConfigRobo:
    # === CONEXÃO ===
    ip: str = "10.1.5.92"
    porta: int = 30004
    
    # === MOVIMENTO BÁSICO ===
    velocidade_padrao: float = 0.1
    aceleracao_padrao: float = 0.1
    altura_segura: float = 0.3
    altura_pegar: float = 0.05
    pausa_entre_movimentos: float = 1.0
    
    # === TIMEOUTS ===
    timeout_movimento: float = 30.0
    timeout_conexao: float = 10.0
    
    # === WORKSPACE SEGURO ===
    workspace_x: Tuple[float, float] = (0.2, 0.8)
    workspace_y: Tuple[float, float] = (-0.4, 0.4)
    workspace_z: Tuple[float, float] = (0.0, 0.5)
    
    # === POSES PREDEFINIDAS ===
    pose_home: List[float] = None
    poses_workspace: dict = None
    
    # === CONFIGURAÇÕES DE SEGURANÇA ===
    nivel_validacao_padrao: str = "advanced"
    estrategia_movimento_padrao: str = "smart_correction"
    habilitar_correcao_automatica: bool = True
    max_tentativas_correcao: int = 3
    distancia_threshold_pontos_intermediarios: float = 0.3
    modo_ultra_seguro: bool = False
    
    # === MOVIMENTO INTELIGENTE ===
    habilitar_correcao_inteligente: bool = True
    habilitar_pontos_intermediarios: bool = True
    distancia_maxima_movimento: float = 1.0
    passo_pontos_intermediarios: float = 0.2
    fator_velocidade_ultra_seguro: float = 0.3
    tentativas_validacao: int = 3
    
    # === BASE DE FERRO ===
    base_ferro_habilitada: bool = True
    altura_base_ferro: float = 0.05
    margem_seguranca_base: float = 0.02
    offset_ombro: float = 0.3

    # === PARÂMETROS DE SEGURANÇA ===
    altura_offset_seguro: float = 0.1
    distancia_minima_cotovelo_tcp: float = 0.028

    # === CONFIGURAÇÕES DE MOVIMENTO ===
    max_mudanca_junta: float = 0.3
    passos_planejamento: int = 10

    # === LIMITES DE WORKSPACE ===
    limites_workspace: dict = None

    # === CONFIGURAÇÕES DE VALIDAÇÃO ===
    habilitar_validacao_seguranca: bool = True
    tentativas_validacao: int = 2
    altura_base_ferro: float = 0.05  # já existe, manter
    margem_seguranca_ombro: float = 0.02


    # === CONFIGURAÇÕES DO ORQUESTRADOR ===
    altura_segura: float = 0.3  # Duplicado, manter consistente  
    altura_pegar: float = 0.05  # Duplicado, manter consistente
    velocidade_normal: float = 0.1  # Usar velocidade_padrao existente
    velocidade_precisa: float = 0.05
    pausa_entre_jogadas: float = 2.0
    auto_calibrar: bool = True
    margem_seguranca_base: float = 0.02  # Renomear de margem_seguranca_base
    validar_antes_executar: bool = True
    modo_logs_limpo: bool = True

    # === CALIBRAÇÃO E TESTES ===
    posicoes_teste_calibracao: List[int] = None
    velocidade_calibracao: float = 0.05


    # === MOVIMENTO INTELIGENTE E CORREÇÃO ===
    max_tentativas_correcao: int = 3  # já existe, manter
    habilitar_correcao_inteligente: bool = True  # renomear de habilitar_correcao_inteligente
    habilitar_pontos_intermediarios: bool = True  # renomear de habilitar_pontos_intermediarios
    passo_pontos_intermediarios: float = 0.2  # já existe, manter
    max_pontos_intermediarios: int = 5
    fator_velocidade_ultra_seguro: float = 0.3  # já existe, manter
    velocidade_movimento_lento: float = 0.02
    aceleracao_movimento_lento: float = 0.02

    # === LIMITES DE VELOCIDADE E SEGURANÇA ===
    velocidade_minima: float = 0.005
    velocidade_maxima: float = 0.2
    aceleracao_minima: float = 0.005  
    aceleracao_maxima: float = 0.2

    # === CONFIGURAÇÕES DE PARADA ===
    desaceleracao_parada: float = 2.0

    CALIBRATION_CONFIG = {
    'velocidade_padrao': 0.1,
    'aceleracao_padrao': 0.2,
    'max_tentativas_correcao': 5,
    'altura_segura_calibracao': 0.42,
    'fator_reducao_y': 0.9
    }

    def __post_init__(self):
        if self.pose_home is None:
            self.pose_home = [0.4, 0.0, 0.4, 0.0, 3.14, 0.0]
            
        if self.poses_workspace is None:
            self.poses_workspace = {
                "center": [0.3, 0.0, 0.2, 0.0, 3.14, 0.0],
                "left": [0.3, 0.3, 0.2, 0.0, 3.14, 0.0],
                "right": [0.3, -0.3, 0.2, 0.0, 3.14, 0.0],
                "front": [0.5, 0.0, 0.2, 0.0, 3.14, 0.0],
                "back": [0.1, 0.0, 0.2, 0.0, 3.14, 0.0]
            }
            
        if self.limites_workspace is None:
            self.limites_workspace = {
                'x_min': -0.8, 'x_max': 0.8,
                'y_min': -0.8, 'y_max': 0.8,
                'z_min': 0.05, 'z_max': 0.8,
                'rx_min': -3.14159, 'rx_max': 3.14159,
                'ry_min': -3.14159, 'ry_max': 3.14159,
                'rz_min': -3.14159, 'rz_max': 3.14159
            }
            
        if self.posicoes_teste_calibracao is None:
            self.posicoes_teste_calibracao = [0, 4, 8]

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