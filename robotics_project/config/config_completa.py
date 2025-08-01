# ARQUIVO: config/config_completa.py
from dataclasses import dataclass
from typing import List, Tuple, Dict
import os
from enum import Enum
import math

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
    
    # === BASE DE FERRO - CONFIGURAÇÕES DETALHADAS ===
    base_ferro_habilitada: bool = True
    altura_base_ferro: float = 0.05
    margem_seguranca_base: float = 0.02
    margem_seguranca_cotovelo: float = 0.02
    margem_seguranca_base_ferro: float = 0.03  # Margem específica para conflitos
    offset_cotovelo: float = 0.3
    aplicar_correcao_base_ferro: bool = True

    # === PARÂMETROS DE SEGURANÇA AVANÇADOS ===
    altura_offset_seguro: float = 0.1
    distancia_minima_cotovelo_tcp: float = 0.028
    elbow_safety_margin: float = 0.02  # Usado no validate_elbow_height_constraint

    # === CONFIGURAÇÕES DE MOVIMENTO AVANÇADAS ===
    max_mudanca_junta: float = 0.3
    passos_planejamento: int = 10
    max_joint_change: float = 0.3  # Alias para max_mudanca_junta
    planning_steps: int = 10       # Alias para passos_planejamento

    # === LIMITES DE WORKSPACE DETALHADOS ===
    limites_workspace: dict = None

    # === CONFIGURAÇÕES DE VALIDAÇÃO ROBUSTA ===
    habilitar_validacao_seguranca: bool = True
    enable_safety_validation: bool = True  # Alias para compatibilidade
    validation_retries: int = 3
    tentativas_validacao: int = 3

    # === CONFIGURAÇÕES DO ORQUESTRADOR ===
    velocidade_normal: float = 0.1  # Usar velocidade_padrao existente
    velocidade_precisa: float = 0.05
    pausa_entre_jogadas: float = 2.0
    auto_calibrar: bool = True
    validar_antes_executar: bool = True
    modo_logs_limpo: bool = True

    # === CALIBRAÇÃO E TESTES ===
    posicoes_teste_calibracao: List[int] = None
    velocidade_calibracao: float = 0.05

    # === MOVIMENTO INTELIGENTE E CORREÇÃO AVANÇADA ===
    max_pontos_intermediarios: int = 5
    velocidade_movimento_lento: float = 0.02
    aceleracao_movimento_lento: float = 0.02

    # === LIMITES DE VELOCIDADE E SEGURANÇA ===
    velocidade_minima: float = 0.005
    velocidade_maxima: float = 0.2
    aceleracao_minima: float = 0.005  
    aceleracao_maxima: float = 0.2

    # === CONFIGURAÇÕES DE PARADA ===
    desaceleracao_parada: float = 2.0

    # === DIAGNÓSTICO E DEBUG AVANÇADO ===
    habilitar_diagnostico_avancado: bool = True
    correcao_baseada_articulacoes: bool = True
    max_tentativas_correcao_articulacoes: int = 5
    
    # === LIMITES ESPECÍFICOS UR ===
    limites_articulacoes: dict = None
    margem_seguranca_articulacoes: float = 0.05  # 5% de margem
    
    # === BASE DE FERRO - CONFIGURAÇÕES MODELO-ESPECÍFICAS ===
    modelo_ur: str = "UR3e"  # UR3e, UR5e, UR10e, etc.
    altura_cotovelo_nominal: float = 0.162  # Altura nominal do cotovelo para UR
    offset_tcp_cotovelo: float = 0.3        # Distância aproximada TCP-cotovelo
    
    # === ESTRATÉGIAS DE CORREÇÃO ===
    estrategias_correcao: List[str] = None
    
    # === DETECÇÃO E CORREÇÃO DE SINGULARIDADES ===
    detectar_singularidades: bool = True
    margem_singularidade_punho: float = 0.1
    margem_singularidade_cotovelo: float = 0.1
    ajuste_orientacao_singularidade: float = 0.05  # radianos
    
    # === VALIDAÇÃO ROBUSTA COM MÚLTIPLAS CONFIGURAÇÕES IK ===
    usar_multiplas_configuracoes_ik: bool = True
    max_configuracoes_ik: int = 8
    
    # === MOVIMENTO COM RECÁLCULO AUTOMÁTICO ===
    habilitar_recalculo_automatico: bool = True
    max_iteracoes_recalculo: int = 3
    
    # === CONFIGURAÇÕES ESPECÍFICAS DO URCONTROLLER ===
    
    # Workspace limits (usado em validate_pose)
    workspace_limits: dict = None  # Será preenchido no __post_init__
    
    # Safety validation settings
    max_movement_distance: float = 1.0  # distancia_maxima_movimento
    
    # Joint validation
    max_joint_change: float = 0.3  # Já definido acima
    
    # Iron base specific
    base_iron_height: float = 0.03  # altura_base_ferro
    
    # Movement correction
    em_movimento: bool = False  # Estado interno do robô
    last_error: str = None      # Último erro registrado
    
    # === CONFIGURAÇÕES DE MOVIMENTO COM PONTOS INTERMEDIÁRIOS ===
    usar_pontos_intermediarios_automatico: bool = True
    numero_pontos_intermediarios_padrao: int = 3
    
    # === CONFIGURAÇÕES DE VALIDAÇÃO ESPECÍFICAS ===
    safe_height_offset: float = 0.1  # altura_offset_seguro
    min_elbow_tcp_distance: float = 0.028  # distancia_minima_cotovelo_tcp
    
    # === CONFIGURAÇÕES DE RETRY E FALLBACK ===
    max_correction_attempts: int = 3  # max_tentativas_correcao
    
    # === CONFIGURAÇÕES PARA DIAGNOSTIC_POSE_REJECTION ===
    habilitar_relatorio_diagnostico: bool = True
    salvar_diagnosticos_em_arquivo: bool = False
    
    # === CONFIGURAÇÕES PARA BENCHMARK E TESTES ===
    executar_benchmark_inicializacao: bool = False
    salvar_resultados_benchmark: bool = False



    def __post_init__(self):
        """Inicialização pós-criação com valores calculados e dependentes"""
        
        # === POSES PREDEFINIDAS ===
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
            
        # === LIMITES DE WORKSPACE (FORMATO URCONTROLLER) ===
        if self.limites_workspace is None:
            self.limites_workspace = {
                'x_min': -0.8, 'x_max': 0.8,
                'y_min': -0.8, 'y_max': 0.8,
                'z_min': 0.05, 'z_max': 0.8,
                'rx_min': -3.14159, 'rx_max': 3.14159,
                'ry_min': -3.14159, 'ry_max': 3.14159,
                'rz_min': -3.14159, 'rz_max': 3.14159
            }
            
        # === WORKSPACE LIMITS (ALIAS PARA COMPATIBILIDADE) ===
        if self.workspace_limits is None:
            self.workspace_limits = self.limites_workspace.copy()
            
        # === POSIÇÕES DE TESTE PARA CALIBRAÇÃO ===
        if self.posicoes_teste_calibracao is None:
            self.posicoes_teste_calibracao = [0, 4, 8]

        # === LIMITES DAS ARTICULAÇÕES BASEADO NO MODELO UR ===
        if self.limites_articulacoes is None:
            base_limits = {

                'UR3e': {
                    'base': (-2*math.pi, 2*math.pi),      # ±360°
                    'shoulder': (-2*math.pi, 2*math.pi),  # ±360°
                    'elbow': (-math.pi, math.pi),         # ±180°
                    'wrist1': (-2*math.pi, 2*math.pi),    # ±360°
                    'wrist2': (-2*math.pi, 2*math.pi),    # ±360°
                    'wrist3': (-2*math.pi, 2*math.pi)     # ±360°
                },
                'UR5e': {
                    'base': (-2*math.pi, 2*math.pi),
                    'shoulder': (-2*math.pi, 2*math.pi),
                    'elbow': (-math.pi, math.pi),
                    'wrist1': (-2*math.pi, 2*math.pi),
                    'wrist2': (-2*math.pi, 2*math.pi),
                    'wrist3': (-2*math.pi, 2*math.pi)
                },
                'UR10e': {
                    'base': (-2*math.pi, 2*math.pi),
                    'shoulder': (-2*math.pi, 2*math.pi),
                    'elbow': (-math.pi, math.pi),
                    'wrist1': (-2*math.pi, 2*math.pi),
                    'wrist2': (-2*math.pi, 2*math.pi),
                    'wrist3': (-2*math.pi, 2*math.pi)
                }
            }
            
            # Usar limites do modelo especificado ou UR5e como padrão
            self.limites_articulacoes = base_limits.get(self.modelo_ur, base_limits['UR3e'])
            
            # Para robôs com base de ferro, aplicar restrições mais conservadoras
            if self.base_ferro_habilitada:
                # Limitar movimento do cotovelo para prevenir conflitos com base
                self.limites_articulacoes['elbow'] = (-1.57, 1.57)  # ±90° mais seguro
                print(f"🔧 Limites de cotovelo restritos para base de ferro: ±90°")
                
        # === ESTRATÉGIAS DE CORREÇÃO EM ORDEM DE PRIORIDADE ===
        if self.estrategias_correcao is None:
            self.estrategias_correcao = [
                "diagnostico_completo",      # Sempre primeiro
                "correcao_base_ferro",       # Prioritário para aplicação com base de ferro
                "correcao_articulacoes",     # Corrigir limites individuais das juntas
                "correcao_singularidades",   # Evitar configurações cinemáticas problemáticas
                "correcao_workspace",        # Correção de workspace tradicional
                "pontos_intermediarios",     # Dividir movimento em etapas
                "movimento_ultra_lento",     # Último recurso com velocidade mínima
            ]
            
        # === AJUSTES ESPECÍFICOS PARA AMBIENTE ===
        self._ajustar_configuracoes_ambiente()
        
        # === VALIDAÇÕES DE SEGURANÇA ===
        self._validar_configuracoes_criticas()
        
        # === CONFIGURAÇÕES DERIVADAS/CALCULADAS ===
        self._calcular_configuracoes_derivadas()

        # =============== CONFIGURAÇÕES DE VALIDAÇÃO ===============
        self.nivel_validacao_padrao = "advanced"  # basic, standard, advanced, complete
        self.estrategia_movimento_padrao = "smart_correction"  # direct, smart_correction, intermediate, ultra_safe
        
        # =============== CONFIGURAÇÕES DE CORREÇÃO AUTOMÁTICA ===============
        self.habilitar_correcao_automatica = True
        self.habilitar_correcao_inteligente = True
        self.max_tentativas_correcao = 3
        self.tentativas_validacao = 2
        
        # =============== CONFIGURAÇÕES DE MOVIMENTO INTELIGENTE ===============
        self.habilitar_pontos_intermediarios = True
        self.distancia_threshold_pontos_intermediarios = 0.3  # metros
        self.distancia_maxima_movimento = 0.5  # metros
        self.passo_pontos_intermediarios = 0.1  # metros entre pontos
        
        # =============== CONFIGURAÇÕES DE SEGURANÇA ===============
        self.modo_ultra_seguro = False
        self.fator_velocidade_ultra_seguro = 0.5  # redução de velocidade em modo ultra-seguro
        
        # =============== CONFIGURAÇÕES DA BASE DE FERRO ===============
        self.base_ferro_habilitada = True
        self.altura_base_ferro = 0.03  # metros - altura da base onde robô está fixado
        self.margem_seguranca_base = 0.02  # metros - margem adicional de segurança
        self.offset_cotovelo = 0.3  # metros - distância estimada TCP->cotovelo
        
        # =============== CONFIGURAÇÕES DE LOGGING ===============
        self.logging_verbose = False  # logs detalhados
        self.logging_summary_only = True  # apenas resumos
        
        # =============== POSES DE WORKSPACE PARA TAPATAN ===============
        self.poses_workspace = {
            # Poses do tabuleiro Tapatan (3x3)
            "tapatan_0": [0.3, 0.2, 0.1, 0.0, 3.14, 0.0],    # posição 0
            "tapatan_1": [0.3, 0.0, 0.1, 0.0, 3.14, 0.0],    # posição 1
            "tapatan_2": [0.3, -0.2, 0.1, 0.0, 3.14, 0.0],   # posição 2
            "tapatan_3": [0.4, 0.2, 0.1, 0.0, 3.14, 0.0],    # posição 3
            "tapatan_4": [0.4, 0.0, 0.1, 0.0, 3.14, 0.0],    # posição 4 (centro)
            "tapatan_5": [0.4, -0.2, 0.1, 0.0, 3.14, 0.0],   # posição 5
            "tapatan_6": [0.5, 0.2, 0.1, 0.0, 3.14, 0.0],    # posição 6
            "tapatan_7": [0.5, 0.0, 0.1, 0.0, 3.14, 0.0],    # posição 7
            "tapatan_8": [0.5, -0.2, 0.1, 0.0, 3.14, 0.0],   # posição 8
            
            # Posições de depósito de peças
            "deposito_jogador1": [0.2, 0.3, 0.1, 0.0, 3.14, 0.0],  # peças do robô
            "deposito_jogador2": [0.2, -0.3, 0.1, 0.0, 3.14, 0.0], # peças removidas
            
            # Posições de segurança
            "observacao": [0.3, 0.0, 0.4, 0.0, 3.14, 0.0],  # posição para observar tabuleiro
            "espera": [0.1, 0.0, 0.3, 0.0, 3.14, 0.0]       # posição de espera
        }
        
        # =============== CONFIGURAÇÕES ESPECÍFICAS TAPATAN ===============
        self.tapatan_config = {
            "altura_tabuleiro": 0.05,  # altura do tabuleiro
            "altura_peca": 0.02,       # altura das peças
            "espacamento_posicoes": 0.1,  # espaçamento entre posições
            "validacao_pre_movimento": True,  # validar antes de cada movimento
            "estrategia_movimento_tapatan": "smart_correction",  # estratégia específica para Tapatan
            "usar_pontos_intermediarios_tapatan": True
        }

    def _ajustar_configuracoes_ambiente(self):
        """Ajusta configurações baseado no ambiente (simulação vs robô real)"""
        
        # Detectar ambiente de simulação
        if self.ip == "127.0.0.1" or "sim" in self.ip.lower() or "localhost" in self.ip.lower():
            print("🔧 Detectado ambiente de SIMULAÇÃO - ajustando configurações")
            
            # Configurações mais permissivas para simulação
            self.margem_seguranca_base_ferro = 0.01
            self.habilitar_diagnostico_avancado = True
            self.executar_benchmark_inicializacao = True
            self.velocidade_maxima = 0.5  # Mais rápido na simulação
            self.max_tentativas_correcao_articulacoes = 3  # Menos tentativas
            
        else:
            print("🔧 Detectado ROBÔ REAL - configurações de segurança máxima")
            
            # Configurações mais conservadoras para robô real
            self.margem_seguranca_base_ferro = 0.05
            self.velocidade_maxima = 0.15  # Mais conservador
            self.max_tentativas_correcao_articulacoes = 5  # Mais tentativas
            self.habilitar_diagnostico_avancado = True  # Sempre ativo para análise

    def _validar_configuracoes_criticas(self):
        """Valida e corrige configurações que podem causar problemas"""
        
        # Validar altura da base de ferro
        if self.altura_base_ferro <= 0:
            print("⚠️ AVISO: altura_base_ferro deve ser > 0, corrigindo para 0.05m")
            self.altura_base_ferro = 0.05
            
        # Validar margens de segurança
        if self.margem_seguranca_base_ferro <= 0:
            print("⚠️ AVISO: margem_seguranca_base_ferro deve ser > 0, corrigindo para 0.02m")
            self.margem_seguranca_base_ferro = 0.02
            
        # Validar limites de velocidade
        if self.velocidade_minima >= self.velocidade_maxima:
            print("⚠️ AVISO: velocidade_minima >= velocidade_maxima, corrigindo")
            self.velocidade_minima = self.velocidade_maxima * 0.1
            
        # Validar workspace
        if self.limites_workspace['z_min'] < 0:
            print("⚠️ AVISO: z_min < 0 pode causar problemas, ajustando para 0.01m")
            self.limites_workspace['z_min'] = 0.01
            
        # Validar altura segura vs altura base ferro
        altura_minima_segura = self.altura_base_ferro + self.margem_seguranca_base_ferro + 0.1
        if self.altura_segura < altura_minima_segura:
            print(f"⚠️ AVISO: altura_segura muito baixa, ajustando para {altura_minima_segura:.3f}m")
            self.altura_segura = altura_minima_segura

    def _calcular_configuracoes_derivadas(self):
        """Calcula configurações que dependem de outras"""
        
        # Calcular altura mínima do TCP baseada na base de ferro
        self.altura_tcp_minima = self.altura_base_ferro + self.margem_seguranca_base_ferro + 0.05
        
        # Calcular velocidade para movimento ultra-seguro
        self.velocidade_ultra_segura = self.velocidade_minima * 2
        
        # Calcular número de pontos intermediários baseado na distância máxima
        self.pontos_intermediarios_auto = max(2, int(self.distancia_maxima_movimento / self.passo_pontos_intermediarios))
        
        # Ajustar timeout baseado na velocidade mínima
        self.timeout_movimento_lento = self.timeout_movimento * 3

    # === MÉTODOS DE UTILIDADE ===
    
    def get_joint_limits_list(self) -> List[Tuple[float, float]]:
        """Retorna limites das articulações como lista de tuplas (para compatibilidade)"""
        return [
            self.limites_articulacoes['base'],
            self.limites_articulacoes['shoulder'],
            self.limites_articulacoes['elbow'],
            self.limites_articulacoes['wrist1'],
            self.limites_articulacoes['wrist2'],
            self.limites_articulacoes['wrist3']
        ]
    
    def get_workspace_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Retorna limites do workspace organizados por eixo"""
        return {
            'x': (self.limites_workspace['x_min'], self.limites_workspace['x_max']),
            'y': (self.limites_workspace['y_min'], self.limites_workspace['y_max']),
            'z': (self.limites_workspace['z_min'], self.limites_workspace['z_max']),
            'rx': (self.limites_workspace['rx_min'], self.limites_workspace['rx_max']),
            'ry': (self.limites_workspace['ry_min'], self.limites_workspace['ry_max']),
            'rz': (self.limites_workspace['rz_min'], self.limites_workspace['rz_max'])
        }
    
    def is_simulation_mode(self) -> bool:
        """Verifica se está em modo simulação"""
        return (self.ip == "127.0.0.1" or 
                "sim" in self.ip.lower() or 
                "localhost" in self.ip.lower())
    
    def get_safe_tcp_height(self) -> float:
        """Retorna altura mínima segura para TCP"""
        return self.altura_base_ferro + self.margem_seguranca_base_ferro + 0.05
    
    def print_configuration_summary(self):
        """Imprime resumo das configurações críticas"""
        print("\n" + "="*60)
        print("🔧 RESUMO DAS CONFIGURAÇÕES DO ROBÔ")
        print("="*60)
        print(f"Modelo UR: {self.modelo_ur}")
        print(f"IP: {self.ip} ({'SIMULAÇÃO' if self.is_simulation_mode() else 'ROBÔ REAL'})")
        print(f"Base de ferro: {'HABILITADA' if self.base_ferro_habilitada else 'DESABILITADA'}")
        if self.base_ferro_habilitada:
            print(f"  Altura base: {self.altura_base_ferro:.3f}m")
            print(f"  Margem segurança: {self.margem_seguranca_base_ferro:.3f}m")
            print(f"  Altura TCP mínima: {self.get_safe_tcp_height():.3f}m")
        print(f"Velocidade: {self.velocidade_minima:.3f} - {self.velocidade_maxima:.3f} m/s")
        print(f"Workspace Z: {self.limites_workspace['z_min']:.3f} - {self.limites_workspace['z_max']:.3f}m")
        print(f"Validação segurança: {'HABILITADA' if self.habilitar_validacao_seguranca else 'DESABILITADA'}")
        print(f"Correção inteligente: {'HABILITADA' if self.habilitar_correcao_inteligente else 'DESABILITADA'}")
        print(f"Diagnóstico avançado: {'HABILITADO' if self.habilitar_diagnostico_avancado else 'DESABILITADO'}")
        print("="*60)


@dataclass  
class ConfigJogo:
    profundidade_ia: int = 3
    debug_mode: bool = False
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

# Função de conveniência para imprimir configurações
def print_all_configurations():
    """Imprime resumo de todas as configurações"""
    CONFIG['robo'].print_configuration_summary()
    print(f"\n🎮 Configurações do jogo:")
    print(f"  Profundidade IA: {CONFIG['jogo'].profundidade_ia}")
    print(f"  Modo debug: {CONFIG['jogo'].modo_debug}")
    print(f"\n💾 Configurações do sistema:")
    print(f"  Pasta logs: {CONFIG['sistema'].pasta_logs}")
    print(f"  Usar câmera real: {CONFIG['sistema'].usar_camera_real}")

# Para uso direto
if __name__ == "__main__":
    print_all_configurations()