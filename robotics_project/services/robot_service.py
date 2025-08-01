import json
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path
import math

from logic_control.ur_controller import URController
from config.config_completa import ConfigRobo
from services.game_service import gerar_tabuleiro_tapatan

class RobotStatus(Enum):
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    MOVING = "moving"
    IDLE = "idle"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"
    VALIDATING = "validating"

class MovementType(Enum):
    LINEAR = "linear"
    JOINT = "joint"
    PICK_PLACE = "pick_place"
    HOME = "home"
    SEQUENCE = "sequence"
    SMART_CORRECTION = "smart_correction"  #  NOVO: Movimento com correção automática
    INTERMEDIATE_POINTS = "intermediate_points"  #  NOVO: Movimento com pontos intermediários

class ValidationLevel(Enum):
    BASIC = "basic"           # Apenas workspace
    STANDARD = "standard"     # Workspace + alcançabilidade  
    ADVANCED = "advanced"     # Workspace + alcançabilidade + UR safety limits
    COMPLETE = "complete"     #  NOVO: Validação completa com todas as verificações

class MovementStrategy(Enum):
    DIRECT = "direct"                    # Movimento direto
    SMART_CORRECTION = "smart_correction"  #  NOVO: Com correção automática
    INTERMEDIATE = "intermediate"        #  NOVO: Com pontos intermediários
    ULTRA_SAFE = "ultra_safe"           #  NOVO: Todas as estratégias de segurança

@dataclass
class RobotPose:
    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float
    
    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z, self.rx, self.ry, self.rz]
    
    @classmethod
    def from_list(cls, pose_list: List[float]):
        if len(pose_list) != 6:
            raise ValueError("Pose deve ter exatamente 6 elementos")
        return cls(*pose_list)
    
    def __str__(self):
        return f"Pose(x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f}, rx={self.rx:.3f}, ry={self.ry:.3f}, rz={self.rz:.3f})"

@dataclass
class MovementCommand:
    type: MovementType
    target_pose: Optional[RobotPose] = None
    speed: Optional[float] = None
    acceleration: Optional[float] = None
    validation_level: ValidationLevel = ValidationLevel.ADVANCED  #  NOVO
    movement_strategy: MovementStrategy = MovementStrategy.SMART_CORRECTION  #  NOVO
    parameters: Optional[Dict[str, Any]] = None

@dataclass
class PickPlaceCommand:
    origin: RobotPose
    destination: RobotPose
    safe_height: float
    pick_height: float
    speed_normal: float = 0.1
    speed_precise: float = 0.05
    validation_level: ValidationLevel = ValidationLevel.COMPLETE  #  NOVO: Validação completa para pick&place

@dataclass
class ValidationResult:
    """ NOVO: Resultado detalhado de validação"""
    is_valid: bool
    workspace_ok: bool = False
    reachability_ok: bool = False
    safety_limits_ok: bool = False
    corrections_applied: List[str] = None
    final_pose: Optional[List[float]] = None
    error_message: Optional[str] = None

class RobotService:
    def __init__(self, config_file: Optional[str] = None):
        # Usar config fornecida ou criar padrão
        self.config_robo = ConfigRobo()
        self.robot_ip = self.config_robo.ip
        self.controller: Optional[URController] = None
        self.status = RobotStatus.DISCONNECTED
        self.last_error: Optional[str] = None
        self.poses = gerar_tabuleiro_tapatan
        
        # Converter ConfigRobo para formato dict compatível (temporário)
        self.config = self._convert_config_to_dict()
        
        # Carregar configuração adicional se fornecida
        if config_file:
            additional_config = self.load_config(config_file)
            self.config.update(additional_config)
        
        # Setup logging
        self.setup_logging()
        
        # Histórico de movimentos para análise
        self.movement_history: List[Dict] = []
        self.validation_stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "corrections_applied": 0,
            "movements_with_intermediate_points": 0
        }
        self.verbose_logging = False
        self.log_summary_only = True
        
    def setup_logging(self):
        """Configura sistema de logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('robot_service.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('RobotService')

    def _convert_config_to_dict(self) -> Dict:
        """Converte ConfigRobo para dict mantendo compatibilidade"""
        return {
            "speed": self.config_robo.velocidade_padrao,
            "acceleration": self.config_robo.aceleracao_padrao,
            "safe_height": self.config_robo.altura_segura,
            "pick_height": self.config_robo.altura_pegar,
            "pause_between_moves": self.config_robo.pausa_entre_movimentos,
            "home_pose": self.config_robo.pose_home,
            
            "default_validation_level": self.config_robo.nivel_validacao_padrao,
            "default_movement_strategy": self.config_robo.estrategia_movimento_padrao,
            "enable_auto_correction": self.config_robo.habilitar_correcao_automatica,
            "max_correction_attempts": self.config_robo.max_tentativas_correcao,
            "intermediate_points_distance_threshold": self.config_robo.distancia_threshold_pontos_intermediarios,
            "ultra_safe_mode": self.config_robo.modo_ultra_seguro,
            
            "smart_movement": {
                "enable_smart_correction": self.config_robo.habilitar_correcao_inteligente,
                "enable_intermediate_points": self.config_robo.habilitar_pontos_intermediarios,
                "max_movement_distance": self.config_robo.distancia_maxima_movimento,
                "intermediate_points_step": self.config_robo.passo_pontos_intermediarios,
                "ultra_safe_speed_factor": self.config_robo.fator_velocidade_ultra_seguro,
                "validation_retries": self.config_robo.tentativas_validacao
            },

            "iron_base_constraint": {
                "enabled": self.config_robo.base_ferro_habilitada,
                "base_height": self.config_robo.altura_base_ferro,
                "safety_margin": self.config_robo.margem_seguranca_base,
                "elbow_offset": self.config_robo.offset_cotovelo
            },
            
        }

    def load_config(self, config_file: str) -> Dict:
        """Carrega configuração de arquivo JSON"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            self.logger.info(f" Configuração carregada de {config_file}")
            return config
        except Exception as e:
            self.logger.error(f" Erro ao carregar configuração: {e}")
            return {}

    def save_config(self, config_file: str):
        """Salva configuração atual em arquivo JSON"""
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            self.logger.info(f" Configuração salva em {config_file}")
        except Exception as e:
            self.logger.error(f" Erro ao salvar configuração: {e}")

    def connect(self) -> bool:
        """Conecta ao robô"""
        try:
            self.logger.info(f" Conectando ao robô em {self.robot_ip}...")
            self.controller = URController(
                robot_ip=self.robot_ip,
                speed=self.config["speed"],
                acceleration=self.config["acceleration"]
            )
            
            if self.controller.is_connected():
                self.status = RobotStatus.CONNECTED
                
                #  NOVO: Configurar parâmetros de segurança no controlador
                if self.config.get("enable_auto_correction", True):
                    self.controller.enable_safety_mode(True)
                
                self.logger.info(" Robô conectado com sucesso")
                self.logger.info(f" Modo de segurança: {'HABILITADO' if self.config.get('enable_auto_correction', True) else 'DESABILITADO'}")
                return True
            else:
                self.status = RobotStatus.ERROR
                self.last_error = "Falha na conexão"
                self.logger.error(" Falha ao conectar com o robô")
                return False
                
        except Exception as e:
            self.status = RobotStatus.ERROR
            self.last_error = str(e)
            self.logger.error(f" Erro ao conectar: {e}")
            return False

    def disconnect(self):
        """Desconecta do robô"""
        try:
            if self.controller:
                self.controller.disconnect()
                self.controller = None
            self.status = RobotStatus.DISCONNECTED
            self.logger.info(" Robô desconectado")
        except Exception as e:
            self.logger.error(f" Erro ao desconectar: {e}")

    # ===================  NOVAS FUNÇÕES DE VALIDAÇÃO ===================

    def validate_pose(self, pose: RobotPose, validation_level: ValidationLevel = None) -> ValidationResult:
        """
         NOVA FUNÇÃO: Validação avançada de pose com níveis configuráveis
        """
        if not self._check_connection():
            return ValidationResult(
                is_valid=False,
                error_message="Robô não conectado"
            )
        
        if validation_level is None:
            validation_level = ValidationLevel(self.config.get("default_validation_level", "advanced"))
        
        self.status = RobotStatus.VALIDATING
        self.validation_stats["total_validations"] += 1
        
        try:
            pose_list = pose.to_list()
            result = ValidationResult(is_valid=False, corrections_applied=[])
            
            # 1. Validação básica de workspace
            result.workspace_ok = self.controller.validate_pose(pose_list)
            
            if validation_level in [ValidationLevel.STANDARD, ValidationLevel.ADVANCED, ValidationLevel.COMPLETE]:
                # 2. Validação de alcançabilidade
                result.reachability_ok = self.controller.validate_pose_reachability(pose_list)
            else:
                result.reachability_ok = True
            
            if validation_level in [ValidationLevel.ADVANCED, ValidationLevel.COMPLETE]:
                # 3.  NOVA: Validação de limites de segurança UR
                result.safety_limits_ok = self.controller.validate_pose_safety_limits(pose_list)
            else:
                result.safety_limits_ok = True
            
            # Resultado final
            result.is_valid = result.workspace_ok and result.reachability_ok and result.safety_limits_ok
            result.final_pose = pose_list
            
            if result.is_valid:
                self.validation_stats["successful_validations"] += 1
                
            self.status = RobotStatus.IDLE
            return result
            
        except Exception as e:
            self.status = RobotStatus.ERROR
            self.last_error = str(e)
            return ValidationResult(
                is_valid=False,
                error_message=f"Erro na validação: {e}"
            )

    def test_pose_validation(self, pose: RobotPose, validation_level: ValidationLevel = ValidationLevel.COMPLETE) -> ValidationResult:
        """
         NOVA FUNÇÃO: Testa validação sem executar movimento
        """
        if not self._check_connection():
            return ValidationResult(is_valid=False, error_message="Robô não conectado")
            
        try:
            self.logger.info(f" Testando validação da pose: {pose}")
            
            # Usar função de teste do URController
            is_valid = self.controller.test_pose_validation(pose.to_list())
            
            return ValidationResult(
                is_valid=is_valid,
                workspace_ok=True,  # Detalhes são mostrados no log do URController
                reachability_ok=True,
                safety_limits_ok=True,
                final_pose=pose.to_list()
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Erro no teste: {e}"
            )
        
    def validate_iron_base_constraint(self, pose: RobotPose) -> bool:
        """
        🔥 NOVA: Validação da base de ferro no nível alto
        """
        if not self.config["iron_base_constraint"]["enabled"]:
            return True
            
        # Estimar posição do cotovelo
        estimated_elbow_z = pose.z - self.config["iron_base_constraint"]["elbow_offset"]
        min_height = (self.config["iron_base_constraint"]["base_height"] + 
                    self.config["iron_base_constraint"]["safety_margin"])
        
        if estimated_elbow_z < min_height:
            if self.verbose_logging:
                self.logger.warning(f"⚠️ Pose rejeitada - cotovelo muito baixo: {estimated_elbow_z:.3f}m")
            return False
        
        return True

    def find_alternative_pose(self, problematic_pose: RobotPose) -> Optional[RobotPose]:
        """
        🔥 NOVA: Encontra pose alternativa quando original é inviável
        """
        if not self.validate_iron_base_constraint(problematic_pose):
            # Elevar TCP para proteger cotovelo
            min_tcp_z = (self.config["iron_base_constraint"]["base_height"] + 
                        self.config["iron_base_constraint"]["safety_margin"] + 
                        self.config["iron_base_constraint"]["elbow_offset"] + 0.05)  # +5cm segurança
            
            if problematic_pose.z < min_tcp_z:
                alternative = RobotPose(
                    x=problematic_pose.x,
                    y=problematic_pose.y,
                    z=min_tcp_z,
                    rx=problematic_pose.rx,
                    ry=problematic_pose.ry,
                    rz=problematic_pose.rz
                )
                
                if self.log_summary_only:
                    self.logger.info(f"Pose corrigida para proteger cotovelo: Z {problematic_pose.z:.3f} -> {min_tcp_z:.3f}")
                
                return alternative
        
        return None
    # ===================  FUNÇÕES DE MOVIMENTO ATUALIZADAS ===================

    def move_to_pose(self, pose: RobotPose, speed: Optional[float] = None, 
                    acceleration: Optional[float] = None,
                    movement_strategy: MovementStrategy = None,
                    validation_level: ValidationLevel = None,
                    enable_intelligent_correction: bool = True) -> bool:
        """
        🔥 FUNÇÃO ATUALIZADA: Move robô com correção inteligente integrada
        Agora usa todas as melhorias do URController
        """
        if not self._check_connection():
            return False
        
        # 🔥 NOVA: Verificar restrição da base de ferro primeiro
        if not self.validate_iron_base_constraint(pose):
            if enable_intelligent_correction:
                self.logger.info(" Pose conflita com base de ferro - tentando correção automática...")
                alternative_pose = self.find_alternative_pose(pose)
                if alternative_pose:
                    pose = alternative_pose
                    self.logger.info(f" Pose corrigida para proteger base de ferro")
                else:
                    self.status = RobotStatus.ERROR
                    self.last_error = "Pose inviável - cotovelo abaixo da base"
                    self.logger.error(" Não foi possível corrigir conflito com base de ferro")
                    return False
            else:
                self.status = RobotStatus.ERROR
                self.last_error = "Pose conflita com base de ferro e correção está desabilitada"
                return False
        
        # Usar configurações padrão se não especificadas
        if movement_strategy is None:
            if enable_intelligent_correction:
                movement_strategy = MovementStrategy.SMART_CORRECTION  # 🔥 NOVO padrão
            else:
                movement_strategy = MovementStrategy.DIRECT
        
        if validation_level is None:
            validation_level = ValidationLevel(self.config.get("default_validation_level", "advanced"))
            
        try:
            self.status = RobotStatus.MOVING
            
            # Log mais informativo
            if self.verbose_logging:
                self.logger.info(f" Movendo para: {pose}")
                self.logger.info(f" Estratégia: {movement_strategy.value}, Validação: {validation_level.value}")
                self.logger.info(f" Correção inteligente: {'HABILITADA' if enable_intelligent_correction else 'DESABILITADA'}")
            else:
                self.logger.info(f" Movimento: {movement_strategy.value}")
            
            # Usar velocidades especificadas ou padrão
            move_speed = speed or self.config["speed"]
            move_acceleration = acceleration or self.config["acceleration"]
            
            # Atualizar parâmetros do controlador se necessário
            if speed or acceleration:
                self.controller.set_speed_parameters(move_speed, move_acceleration)
            
            # Registrar movimento no histórico
            movement_record = {
                "timestamp": time.time(),
                "target_pose": asdict(pose),
                "strategy": movement_strategy.value,
                "validation_level": validation_level.value,
                "intelligent_correction": enable_intelligent_correction,
                "speed": move_speed,
                "acceleration": move_acceleration
            }
            
            success = False
            pose_list = pose.to_list()
            
            # 🔥 EXECUTAR ESTRATÉGIA DE MOVIMENTO ATUALIZADA
            if movement_strategy == MovementStrategy.DIRECT:
                # Movimento direto (modo legado)
                success = self.controller.move_to_pose_safe(
                    pose_list, move_speed, move_acceleration, 
                    use_smart_correction=False
                )
                
            elif movement_strategy == MovementStrategy.SMART_CORRECTION:
                # 🔥 NOVA: Movimento com correção automática inteligente
                success, corrected_pose = self.controller.move_to_pose_with_smart_correction(
                    pose_list, move_speed, move_acceleration, 
                    max_correction_attempts=self.config.get("max_correction_attempts", 3)
                )
                if corrected_pose:
                    movement_record["corrected_pose"] = corrected_pose
                    self.validation_stats["corrections_applied"] += 1
                    if self.verbose_logging:
                        self.logger.info(" Correções automáticas aplicadas")
                        
            elif movement_strategy == MovementStrategy.INTERMEDIATE:
                # 🔥 NOVA: Movimento com pontos intermediários
                current_pose = self.controller.get_current_pose()
                if current_pose:
                    distance = math.sqrt(
                        (pose_list[0] - current_pose[0])**2 +
                        (pose_list[1] - current_pose[1])**2 +
                        (pose_list[2] - current_pose[2])**2
                    )
                    
                    step_distance = self.config["smart_movement"]["intermediate_points_step"]
                    num_points = max(2, int(distance / step_distance))
                    num_points = min(num_points, 5)  # Máximo 5 pontos
                    
                    success = self.controller.move_with_intermediate_points(
                        pose_list, move_speed, move_acceleration, num_points
                    )
                    if success:
                        self.validation_stats["movements_with_intermediate_points"] += 1
                        movement_record["intermediate_points"] = num_points
                        if self.verbose_logging:
                            self.logger.info(f"🚀 Movimento executado com {num_points} pontos intermediários")
                            
            elif movement_strategy == MovementStrategy.ULTRA_SAFE:
                # 🔥 NOVA: Estratégia ultra-segura com todas as validações
                success = self.controller.move_to_pose_safe(
                    pose_list, move_speed, move_acceleration, 
                    use_smart_correction=True
                )
            
            # Registrar resultado
            movement_record["success"] = success
            movement_record["duration"] = time.time() - movement_record["timestamp"]
            self.movement_history.append(movement_record)
            
            # Limitar histórico a 100 movimentos
            if len(self.movement_history) > 100:
                self.movement_history.pop(0)
            
            if success:
                self.status = RobotStatus.IDLE
                if self.log_summary_only:
                    self.logger.info(f" Movimento concluído - {movement_strategy.value}")
                else:
                    # Verificar precisão se verbose
                    final_pose = self.get_current_pose()
                    if final_pose and self.verbose_logging:
                        distance = math.sqrt(
                            (pose.x - final_pose.x)**2 +
                            (pose.y - final_pose.y)**2 +
                            (pose.z - final_pose.z)**2
                        )
                        self.logger.info(f" Precisão do movimento: {distance*1000:.1f}mm")
                return True
            else:
                self.status = RobotStatus.ERROR
                self.last_error = f"Falha no movimento - {movement_strategy.value}"
                self.logger.error(f" Movimento falhou - {movement_strategy.value}")
                
                # 🔥 NOVA: Sugestão automática se falha
                if enable_intelligent_correction and movement_strategy != MovementStrategy.ULTRA_SAFE:
                    self.logger.info(" SUGESTÃO: Tente usar MovementStrategy.ULTRA_SAFE ou debug_pose_detailed()")
                
                return False
                
        except Exception as e:
            self.status = RobotStatus.ERROR
            self.last_error = str(e)
            self.logger.error(f" Erro durante movimento: {e}")
            return False

    def move_home(self) -> bool:
        """Move robô para posição home com segurança máxima"""
        home_pose = RobotPose.from_list(self.config["home_pose"])
        self.logger.info(" Movendo para posição home")
        
        # Home sempre usa estratégia ultra-segura
        return self.move_to_pose(
            home_pose, 
            movement_strategy=MovementStrategy.ULTRA_SAFE,
            validation_level=ValidationLevel.COMPLETE
        )

    def pick_and_place(self, pick_place_cmd: PickPlaceCommand) -> bool:
        """
         FUNÇÃO ATUALIZADA: Pick and place com validação completa em cada etapa
        """
        if not self._check_connection():
            return False
            
        try:
            self.status = RobotStatus.MOVING
            self.logger.info(f" Iniciando pick and place AVANÇADO:")
            self.logger.info(f"     Origem: {pick_place_cmd.origin}")
            self.logger.info(f"     Destino: {pick_place_cmd.destination}")
            self.logger.info(f"     Validação: {pick_place_cmd.validation_level.value}")
            
            #  USAR FUNÇÃO ATUALIZADA DO URCONTROLLER
            success = self.controller.executar_movimento_peca(
                pick_place_cmd.origin.to_list(),
                pick_place_cmd.destination.to_list(),
                pick_place_cmd.safe_height,
                pick_place_cmd.pick_height
            )
            
            if success:
                self.status = RobotStatus.IDLE
                self.logger.info(" Pick and place concluído")
                return True
            else:
                self.status = RobotStatus.ERROR
                self.last_error = "Falha no pick and place"
                self.logger.error(" Falha no pick and place")
                return False
                
        except Exception as e:
            self.status = RobotStatus.ERROR
            self.last_error = str(e)
            self.logger.error(f" Erro durante pick and place: {e}")
            return False

    def correct_pose_with_diagnostics(self, pose: RobotPose) -> Dict[str, Any]:
        """
        🔥 NOVA FUNÇÃO: Corrige pose com diagnóstico completo
        Retorna informações detalhadas sobre o que foi corrigido
        """
        if not self._check_connection():
            return {"success": False, "error": "Robô não conectado"}
        
        try:
            self.logger.info(f"🔧 Iniciando correção inteligente para: {pose}")
            
            # 1. Diagnóstico completo da pose original
            pose_list = pose.to_list()
            diagnostics = self.controller.diagnostic_pose_rejection(pose_list)
            
            # 2. Aplicar correção automática
            corrected_pose_list = self.controller.correct_pose_automatically(pose_list)
            
            # 3. Validar pose corrigida
            is_corrected_valid = False
            if corrected_pose_list:
                is_corrected_valid = self.controller.validate_pose_complete(corrected_pose_list)
            
            result = {
                "success": is_corrected_valid,
                "original_pose": asdict(pose),
                "corrected_pose": RobotPose.from_list(corrected_pose_list).to_dict() if corrected_pose_list else None,
                "diagnostics": diagnostics,
                "corrections_applied": diagnostics.get('sugestoes_correcao', []),
                "validation_passed": is_corrected_valid,
                "error_details": diagnostics.get('joints_problematicas', [])
            }
            
            if is_corrected_valid:
                self.logger.info(" Correção inteligente CONCLUÍDA com sucesso!")
            else:
                self.logger.error(" Correção inteligente FALHOU")
                
            return result
            
        except Exception as e:
            self.logger.error(f" Erro na correção inteligente: {e}")
            return {"success": False, "error": str(e)}

    def move_with_intelligent_correction(self, pose: RobotPose, 
                                    speed: Optional[float] = None,
                                    acceleration: Optional[float] = None,
                                    max_attempts: int = 3) -> Dict[str, Any]:
        """
        🔥 NOVA FUNÇÃO: Movimento com correção inteligente e relatório detalhado
        Substitui o move_to_pose quando você quer detalhes da correção
        """
        if not self._check_connection():
            return {"success": False, "error": "Robô não conectado"}
        
        start_time = time.time()
        movement_log = {
            "original_pose": asdict(pose),
            "attempts": [],
            "final_result": {},
            "total_time": 0,
            "corrections_applied": []
        }
        
        try:
            self.status = RobotStatus.MOVING
            
            # Usar velocidades especificadas ou padrão
            move_speed = speed or self.config["speed"]
            move_acceleration = acceleration or self.config["acceleration"]
            
            self.logger.info(f" Movimento INTELIGENTE iniciado para: {pose}")
            
            # Usar a nova função do URController
            success, final_pose = self.controller.move_to_pose_with_smart_correction(
                pose.to_list(), move_speed, move_acceleration, max_attempts
            )
            
            movement_log["final_result"] = {
                "success": success,
                "final_pose": RobotPose.from_list(final_pose).to_dict() if final_pose else None,
                "execution_time": time.time() - start_time
            }
            
            if success:
                self.status = RobotStatus.IDLE
                self.logger.info(" Movimento inteligente CONCLUÍDO!")
                return {"success": True, "details": movement_log}
            else:
                self.status = RobotStatus.ERROR
                self.last_error = "Movimento inteligente falhou"
                self.logger.error(" Movimento inteligente FALHOU após todas as tentativas")
                return {"success": False, "details": movement_log, "error": "Todas as estratégias falharam"}
                
        except Exception as e:
            self.status = RobotStatus.ERROR
            self.last_error = str(e)
            self.logger.error(f" Erro no movimento inteligente: {e}")
            return {"success": False, "error": str(e)}

    def test_and_fix_calibration_poses(self, poses: List[RobotPose]) -> Dict[str, Any]:
        """
        🔧 NOVA FUNÇÃO: Testa e corrige poses de calibração
        Ideal para debugging de sequências de calibração
        """
        if not self._check_connection():
            return {"error": "Robô não conectado"}
        
        self.logger.info(f" Testando e corrigindo {len(poses)} poses de calibração")
        
        results = {
            "total_poses": len(poses),
            "original_valid": 0,
            "corrected_valid": 0,
            "impossible_poses": 0,
            "pose_details": [],
            "recommendations": []
        }
        
        for i, pose in enumerate(poses):
            self.logger.info(f" Analisando pose {i+1}/{len(poses)}")
            
            pose_result = {
                "index": i,
                "original_pose": asdict(pose),
                "is_original_valid": False,
                "corrected_pose": None,
                "is_corrected_valid": False,
                "corrections_needed": [],
                "status": "unknown"
            }
            
            try:
                # 1. Testar pose original
                pose_list = pose.to_list()
                pose_result["is_original_valid"] = self.controller.validate_pose_complete(pose_list)
                
                if pose_result["is_original_valid"]:
                    results["original_valid"] += 1
                    pose_result["status"] = " VÁLIDA"
                else:
                    # 2. Tentar correção usando a nova função específica
                    corrected_pose_list, correction_success = self.controller.fix_calibration_pose(i, pose_list)
                    
                    if correction_success:
                        pose_result["corrected_pose"] = RobotPose.from_list(corrected_pose_list).to_dict()
                        pose_result["is_corrected_valid"] = True
                        results["corrected_valid"] += 1
                        pose_result["status"] = " CORRIGIDA"
                    else:
                        results["impossible_poses"] += 1
                        pose_result["status"] = " IMPOSSÍVEL"
                
            except Exception as e:
                pose_result["status"] = f" ERRO: {str(e)}"
                results["impossible_poses"] += 1
            
            results["pose_details"].append(pose_result)
        
        # Gerar recomendações
        success_rate = ((results["original_valid"] + results["corrected_valid"]) / results["total_poses"]) * 100
        
        if success_rate < 80:
            results["recommendations"].append("Taxa de sucesso baixa - verificar altura da base de ferro")
        if results["corrected_valid"] > results["original_valid"]:
            results["recommendations"].append("Muitas correções necessárias - considerar ajustar poses originais")
        if results["impossible_poses"] > 2:
            results["recommendations"].append("Poses impossíveis detectadas - verificar limites do workspace")
        
        self.logger.info(f" RESULTADO: {results['original_valid']} válidas, {results['corrected_valid']} corrigidas, {results['impossible_poses']} impossíveis")
        
        return results

    # ===================  NOVAS FUNÇÕES DE SEQUÊNCIA ===================

    def execute_sequence_advanced(self, commands: List[MovementCommand], 
                                 stop_on_failure: bool = True,
                                 validation_summary: bool = True) -> Dict[str, Any]:
        """
         NOVA FUNÇÃO: Execução avançada de sequência com relatório detalhado
        """
        if not self._check_connection():
            return {"success": False, "error": "Robô não conectado"}
        
        self.logger.info(f" Executando sequência AVANÇADA de {len(commands)} comandos")
        
        sequence_result = {
            "success": False,
            "total_commands": len(commands),
            "executed_commands": 0,
            "failed_commands": 0,
            "validations_performed": 0,
            "corrections_applied": 0,
            "intermediate_movements": 0,
            "execution_time": 0,
            "command_results": [],
            "error_summary": []
        }
        
        start_time = time.time()
        
        for i, cmd in enumerate(commands):
            self.logger.info(f" Executando comando {i+1}/{len(commands)}: {cmd.type.value}")
            
            command_start = time.time()
            command_result = {
                "index": i,
                "type": cmd.type.value,
                "success": False,
                "execution_time": 0,
                "validation_level": cmd.validation_level.value if hasattr(cmd, 'validation_level') else "standard",
                "movement_strategy": cmd.movement_strategy.value if hasattr(cmd, 'movement_strategy') else "direct"
            }
            
            try:
                if cmd.type == MovementType.LINEAR:
                    if cmd.target_pose:
                        # Validar antes de executar se solicitado      
                        success = self.move_to_pose(
                            cmd.target_pose, 
                            cmd.speed, 
                            cmd.acceleration,
                            cmd.movement_strategy,
                            cmd.validation_level
                        )
                        command_result["success"] = success
                        
                elif cmd.type == MovementType.HOME:
                    success = self.move_home()
                    command_result["success"] = success
                    
                elif cmd.type == MovementType.PICK_PLACE:
                    if cmd.parameters:
                        pick_place_cmd = PickPlaceCommand(
                            origin=RobotPose.from_list(cmd.parameters["origin"]),
                            destination=RobotPose.from_list(cmd.parameters["destination"]),
                            safe_height=cmd.parameters.get("safe_height", self.config["safe_height"]),
                            pick_height=cmd.parameters.get("pick_height", self.config["pick_height"]),
                            validation_level=cmd.validation_level
                        )
                        success = self.pick_and_place(pick_place_cmd)
                        command_result["success"] = success
                
                command_result["execution_time"] = time.time() - command_start
                
                if command_result["success"]:
                    sequence_result["executed_commands"] += 1
                else:
                    sequence_result["failed_commands"] += 1
                    if stop_on_failure:
                        self.logger.error(f" Falha no comando {i+1} - sequência interrompida")
                        sequence_result["error_summary"].append(f"Comando {i+1}: Execução falhou")
                        break
                
            except Exception as e:
                command_result["error"] = str(e)
                sequence_result["failed_commands"] += 1
                sequence_result["error_summary"].append(f"Comando {i+1}: {str(e)}")
                
                if stop_on_failure:
                    break
            
            sequence_result["command_results"].append(command_result)
            
            # Pausa entre comandos
            time.sleep(self.config["pause_between_moves"])
        
        sequence_result["execution_time"] = time.time() - start_time
        sequence_result["success"] = sequence_result["failed_commands"] == 0
        
        # Estatísticas do histórico atual
        sequence_result["corrections_applied"] = self.validation_stats["corrections_applied"]
        sequence_result["intermediate_movements"] = self.validation_stats["movements_with_intermediate_points"]
        
        self.logger.info(f" Sequência concluída:")
        self.logger.info(f"    Sucessos: {sequence_result['executed_commands']}")
        self.logger.info(f"    Falhas: {sequence_result['failed_commands']}")
        self.logger.info(f"    Tempo total: {sequence_result['execution_time']:.1f}s")
        
        return sequence_result

    def execute_sequence(self, commands: List[MovementCommand]) -> bool:
        """Executa sequência de comandos (interface compatível)"""
        result = self.execute_sequence_advanced(commands, stop_on_failure=True, validation_summary=False)
        return result["success"]

    # ===================  NOVAS FUNÇÕES DE DEBUG E ANÁLISE ===================

    def debug_pose_sequence(self, poses: List[RobotPose], test_only: bool = True) -> Dict[str, Any]:
        """
         NOVA FUNÇÃO: Debug de sequência de poses
        """
        if not self._check_connection():
            return {"error": "Robô não conectado"}
            
        self.logger.info(f" DEBUG: Analisando sequência de {len(poses)} poses")
        
        # Usar função de debug do URController
        poses_list = [pose.to_list() for pose in poses]
        results = self.controller.debug_movement_sequence(poses_list, test_only=test_only)
        
        debug_summary = {
            "total_poses": len(poses),
            "valid_poses": sum(results),
            "invalid_poses": len(results) - sum(results),
            "success_rate": (sum(results) / len(results)) * 100 if results else 0,
            "pose_results": []
        }
        
        for i, (pose, result) in enumerate(zip(poses, results)):
            debug_summary["pose_results"].append({
                "index": i,
                "pose": asdict(pose),
                "valid": result
            })
        
        return debug_summary

    def get_movement_statistics(self) -> Dict[str, Any]:
        """
         NOVA FUNÇÃO: Estatísticas de movimento e validação
        """
        total_movements = len(self.movement_history)
        successful_movements = sum(1 for m in self.movement_history if m.get("success", False))
        
        stats = {
            "total_movements": total_movements,
            "successful_movements": successful_movements,
            "failed_movements": total_movements - successful_movements,
            "success_rate": (successful_movements / total_movements * 100) if total_movements > 0 else 0,
            "validation_stats": self.validation_stats.copy(),
            "strategy_usage": {},
            "average_execution_time": 0
        }
        
        if self.movement_history:
            # Análise de estratégias usadas
            for movement in self.movement_history:
                strategy = movement.get("strategy", "unknown")
                stats["strategy_usage"][strategy] = stats["strategy_usage"].get(strategy, 0) + 1
            
            # Tempo médio de execução
            total_time = sum(m.get("duration", 0) for m in self.movement_history)
            stats["average_execution_time"] = total_time / len(self.movement_history)
        
        return stats

    def get_current_pose(self) -> Optional[RobotPose]:
        """Obtém pose atual do robô"""
        if not self._check_connection():
            return None
            
        try:
            pose_list = self.controller.get_current_pose()
            if pose_list:
                return RobotPose.from_list(pose_list)
            return None
        except Exception as e:
            self.logger.error(f" Erro ao obter pose atual: {e}")
            return None

    def get_status(self) -> Dict[str, Any]:
        """
         FUNÇÃO ATUALIZADA: Status completo com novas informações
        """
        status_dict = {
            "status": self.status.value,
            "connected": self.status not in [RobotStatus.DISCONNECTED, RobotStatus.ERROR],
            "last_error": self.last_error,
            "current_pose": None,
            "robot_details": None,
            "movement_statistics": self.get_movement_statistics(),  #  NOVO
            "safety_configuration": {  #  NOVO
                "validation_level": self.config.get("default_validation_level", "advanced"),
                "movement_strategy": self.config.get("default_movement_strategy", "smart_correction"),
                "auto_correction_enabled": self.config.get("enable_auto_correction", True),
                "ultra_safe_mode": self.config.get("ultra_safe_mode", False),
                "max_correction_attempts": self.config.get("max_correction_attempts", 3)
            }
        }
        
        if self.controller and self.controller.is_connected():
            try:
                current_pose = self.get_current_pose()
                if current_pose:
                    status_dict["current_pose"] = {
                        "x": current_pose.x,
                        "y": current_pose.y,
                        "z": current_pose.z,
                        "rx": current_pose.rx,
                        "ry": current_pose.ry,
                        "rz": current_pose.rz
                    }
                
                robot_status = self.controller.get_robot_status()
                status_dict["robot_details"] = robot_status
                
            except Exception as e:
                self.logger.error(f" Erro ao obter status detalhado: {e}")
        
        return status_dict

    # ===================  NOVAS FUNÇÕES DE CONTROLE DE SEGURANÇA ===================

    def enable_ultra_safe_mode(self, enable: bool = True):
        """
         NOVA FUNÇÃO: Liga/desliga modo ultra-seguro
        """
        self.config["ultra_safe_mode"] = enable
        if self.controller:
            self.controller.enable_safety_mode(enable)
        
        mode_status = "HABILITADO" if enable else "DESABILITADO"
        self.logger.info(f" Modo ultra-seguro {mode_status}")

    def set_validation_level(self, level: ValidationLevel):
        """
         NOVA FUNÇÃO: Define nível de validação padrão
        """
        self.config["default_validation_level"] = level.value
        self.logger.info(f"🔍 Nível de validação definido para: {level.value}")

    def set_movement_strategy(self, strategy: MovementStrategy):
        """
         NOVA FUNÇÃO: Define estratégia de movimento padrão
        """
        self.config["default_movement_strategy"] = strategy.value
        self.logger.info(f" Estratégia de movimento definida para: {strategy.value}")

    def correct_pose_automatically(self, pose: RobotPose) -> Optional[RobotPose]:
        """
         NOVA FUNÇÃO: Corrige pose automaticamente usando URController
        """
        if not self._check_connection():
            return None
            
        try:
            corrected_pose_list = self.controller.correct_pose_automatically(pose.to_list())
            if corrected_pose_list:
                return RobotPose.from_list(corrected_pose_list)
            return None
        except Exception as e:
            self.logger.error(f" Erro na correção automática: {e}")
            return None

    # ===================  FUNÇÕES DE PLANEJAMENTO AVANÇADO ===================

    def plan_safe_trajectory(self, target_pose: RobotPose, 
                           max_intermediate_points: int = 5) -> List[RobotPose]:
        """
         NOVA FUNÇÃO: Planeja trajetória segura com pontos intermediários
        """
        if not self._check_connection():
            return []
            
        try:
            current_pose = self.get_current_pose()
            if not current_pose:
                return []
            
            current_list = current_pose.to_list()
            target_list = target_pose.to_list()
            
            # Calcular distância
            distance = math.sqrt(
                (target_list[0] - current_list[0])**2 +
                (target_list[1] - current_list[1])**2 +
                (target_list[2] - current_list[2])**2
            )
            
            # Determinar número de pontos intermediários
            step_distance = self.config["smart_movement"]["intermediate_points_step"]
            num_points = min(max_intermediate_points, max(1, int(distance / step_distance)))
            
            trajectory = []
            
            # Gerar pontos intermediários
            for i in range(1, num_points + 1):
                factor = i / (num_points + 1)
                
                intermediate_pose_list = [
                    current_list[j] + (target_list[j] - current_list[j]) * factor
                    for j in range(6)
                ]
                
                # Validar ponto intermediário
                if self.controller.validate_pose_complete(intermediate_pose_list):
                    trajectory.append(RobotPose.from_list(intermediate_pose_list))
                else:
                    # Tentar corrigir ponto intermediário
                    corrected = self.controller.correct_pose_automatically(intermediate_pose_list)
                    if corrected and self.controller.validate_pose_complete(corrected):
                        trajectory.append(RobotPose.from_list(corrected))
                        self.logger.info(f" Ponto intermediário {i} corrigido")
                    else:
                        self.logger.warning(f"⚠️ Ponto intermediário {i} rejeitado")
            
            # Adicionar pose final
            trajectory.append(target_pose)
            
            self.logger.info(f" Trajetória planejada com {len(trajectory)} pontos")
            return trajectory
            
        except Exception as e:
            self.logger.error(f" Erro no planejamento de trajetória: {e}")
            return []

    def execute_planned_trajectory(self, trajectory: List[RobotPose], 
                                 speed: Optional[float] = None,
                                 acceleration: Optional[float] = None) -> bool:
        """
         NOVA FUNÇÃO: Executa trajetória planejada
        """
        if not trajectory:
            self.logger.error(" Trajetória vazia")
            return False
            
        self.logger.info(f" Executando trajetória planejada com {len(trajectory)} pontos")
        
        for i, pose in enumerate(trajectory):
            self.logger.info(f" Executando ponto {i+1}/{len(trajectory)}")
            
            success = self.move_to_pose(
                pose, 
                speed, 
                acceleration,
                movement_strategy=MovementStrategy.SMART_CORRECTION,
                validation_level=ValidationLevel.ADVANCED
            )
            
            if not success:
                self.logger.error(f" Falha no ponto {i+1} - trajetória interrompida")
                return False
        
        self.logger.info(" Trajetória executada com sucesso!")
        return True

    # =================== FUNÇÕES DE CONTROLE EXISTENTES ATUALIZADAS ===================

    def emergency_stop(self) -> bool:
        """Parada de emergência"""
        try:
            if self.controller:
                success = self.controller.emergency_stop()
                if success:
                    self.status = RobotStatus.EMERGENCY_STOP
                    self.logger.warning(" PARADA DE EMERGÊNCIA ATIVADA")
                    return True
            return False
        except Exception as e:
            self.logger.error(f" Erro na parada de emergência: {e}")
            return False

    def stop_movement(self) -> bool:
        """Para movimento atual"""
        try:
            if self.controller:
                success = self.controller.stop()
                if success:
                    self.status = RobotStatus.IDLE
                    self.logger.info(" Movimento parado")
                    return True
            return False
        except Exception as e:
            self.logger.error(f" Erro ao parar movimento: {e}")
            return False

    def get_predefined_pose(self, pose_name: str) -> Optional[RobotPose]:
        """Obtém pose predefinida por nome"""
        if pose_name == "home":
            return RobotPose.from_list(self.config["home_pose"])
        elif pose_name in self.poses:
            return RobotPose.from_list(self.poses)
        else:
            self.logger.error(f" Pose '{pose_name}' não encontrada")
            return None

    def update_config(self, new_config: Dict[str, Any]):
        """
         FUNÇÃO ATUALIZADA: Atualiza configuração com validação
        """
        old_config = self.config.copy()
        self.config.update(new_config)
        
        # Validar configurações críticas
        if "default_validation_level" in new_config:
            try:
                ValidationLevel(new_config["default_validation_level"])
            except ValueError:
                self.logger.error(f" Nível de validação inválido: {new_config['default_validation_level']}")
                self.config["default_validation_level"] = old_config["default_validation_level"]
        
        if "default_movement_strategy" in new_config:
            try:
                MovementStrategy(new_config["default_movement_strategy"])
            except ValueError:
                self.logger.error(f" Estratégia de movimento inválida: {new_config['default_movement_strategy']}")
                self.config["default_movement_strategy"] = old_config["default_movement_strategy"]
        
        self.logger.info(" Configuração atualizada")
        
        # Atualizar parâmetros do controlador se conectado
        if self.controller:
            if "speed" in new_config or "acceleration" in new_config:
                self.controller.set_speed_parameters(
                    self.config["speed"],
                    self.config["acceleration"]
                )
            
            if "enable_auto_correction" in new_config:
                self.controller.enable_safety_mode(new_config["enable_auto_correction"])

    def _check_connection(self) -> bool:
        """Verifica se está conectado ao robô"""
        if not self.controller or not self.controller.is_connected():
            self.status = RobotStatus.DISCONNECTED
            self.last_error = "Robô não conectado"
            self.logger.error(" Robô não está conectado")
            return False
        return True

    # ===================  NOVAS FUNÇÕES DE RELATÓRIO ===================

    def generate_safety_report(self) -> Dict[str, Any]:
        """
         NOVA FUNÇÃO: Gera relatório de segurança detalhado
        """
        stats = self.get_movement_statistics()
        
        report = {
            "timestamp": time.time(),
            "robot_status": self.status.value,
            "safety_configuration": {
                "validation_level": self.config.get("default_validation_level"),
                "movement_strategy": self.config.get("default_movement_strategy"),
                "auto_correction_enabled": self.config.get("enable_auto_correction"),
                "ultra_safe_mode": self.config.get("ultra_safe_mode")
            },
            "performance_metrics": {
                "total_movements": stats["total_movements"],
                "success_rate": stats["success_rate"],
                "correction_rate": (stats["validation_stats"]["corrections_applied"] / 
                                  max(stats["total_movements"], 1)) * 100,
                "intermediate_movement_rate": (stats["validation_stats"]["movements_with_intermediate_points"] / 
                                             max(stats["total_movements"], 1)) * 100,
                "average_execution_time": stats["average_execution_time"]
            },
            "validation_statistics": stats["validation_stats"],
            "strategy_distribution": stats["strategy_usage"],
            "recommendations": []
        }
        
        # Gerar recomendações baseadas nos dados
        if report["performance_metrics"]["success_rate"] < 90:
            report["recommendations"].append("Considere usar ValidationLevel.COMPLETE para maior segurança")
        
        if report["performance_metrics"]["correction_rate"] > 20:
            report["recommendations"].append("Alta taxa de correções - verifique configuração do workspace")
        
        if report["performance_metrics"]["average_execution_time"] > 10:
            report["recommendations"].append("Tempo de execução alto - considere otimizar trajetórias")
        
        return report

    def export_movement_history(self, filename: str = None) -> str:
        """
         NOVA FUNÇÃO: Exporta histórico de movimentos para JSON
        """
        if filename is None:
            filename = f"movement_history_{int(time.time())}.json"
        
        export_data = {
            "export_timestamp": time.time(),
            "robot_ip": self.robot_ip,
            "config": self.config,
            "movement_history": self.movement_history,
            "validation_stats": self.validation_stats,
            "safety_report": self.generate_safety_report()
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f" Histórico exportado para: {filename}")
            return filename
        except Exception as e:
            self.logger.error(f" Erro ao exportar histórico: {e}")
            return ""

    def reset_statistics(self):
        """
         NOVA FUNÇÃO: Reseta estatísticas de movimento
        """
        self.movement_history.clear()
        self.validation_stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "corrections_applied": 0,
            "movements_with_intermediate_points": 0
        }
        self.logger.info(" Estatísticas resetadas")

    def set_logging_mode(self, verbose: bool = False, summary_only: bool = True):
        """
        🔥 NOVA: Controla modo de logging
        """
        self.verbose_logging = verbose
        self.log_summary_only = summary_only
        
        # Configurar também no URController
        if self.controller:
            # Desabilitar prints excessivos do URController se possível
            pass
        
        mode = "VERBOSE" if verbose else "RESUMO" if summary_only else "NORMAL"
        self.logger.info(f"🔧 Modo de logging: {mode}")

    def configure_iron_base(self, height: float, margin: float = 0.02, elbow_offset: float = 0.3):
        """
        🔥 NOVA: Configura parâmetros da base de ferro
        """
        self.config["iron_base_constraint"].update({
            "base_height": height,
            "safety_margin": margin,
            "elbow_offset": elbow_offset
        })
        
        # Também configurar no URController
        if self.controller:
            self.controller.set_iron_base_height(height)
        
        self.logger.info(f"🔧 Base de ferro configurada: {height:.3f}m + {margin:.3f}m margem")

    def debug_pose_detailed(self, pose: RobotPose) -> Dict[str, Any]:
        """
        🔍 NOVA FUNÇÃO: Debug detalhado de uma pose específica
        Mostra exatamente por que uma pose falha
        """
        if not self._check_connection():
            return {"error": "Robô não conectado"}
        
        self.logger.info(f"🔍 DEBUG DETALHADO da pose: {pose}")
        
        try:
            pose_list = pose.to_list()
            
            # Usar a nova função de diagnóstico do URController
            diagnostics = self.controller.diagnostic_pose_rejection(pose_list)
            
            # Adicionar informações do RobotService
            debug_info = {
                "pose_analysis": diagnostics,
                "robot_status": self.get_status(),
                "current_pose": self.get_current_pose().to_dict() if self.get_current_pose() else None,
                "workspace_limits": self.controller.workspace_limits if hasattr(self.controller, 'workspace_limits') else None,
                "iron_base_config": self.config.get("iron_base_constraint", {}),
                "safety_settings": {
                    "validation_level": self.config.get("default_validation_level"),
                    "movement_strategy": self.config.get("default_movement_strategy"),
                    "auto_correction": self.config.get("enable_auto_correction"),
                }
            }
            
            # Calcular distância da pose atual
            current_pose = self.get_current_pose()
            if current_pose:
                distance = math.sqrt(
                    (pose.x - current_pose.x)**2 +
                    (pose.y - current_pose.y)**2 +
                    (pose.z - current_pose.z)**2
                )
                debug_info["movement_distance"] = distance
                
                if distance > self.config.get("smart_movement", {}).get("max_movement_distance", 0.5):
                    debug_info["warnings"] = ["Movimento muito grande - considere pontos intermediários"]
            
            self.logger.info("🔍 Debug detalhado concluído - verifique o retorno para análise completa")
            return debug_info
            
        except Exception as e:
            self.logger.error(f" Erro no debug detalhado: {e}")
            return {"error": str(e)}

    def debug_calibration_sequence(self, poses: List[RobotPose], 
                                fix_problems: bool = False) -> Dict[str, Any]:
        """
        🧪 NOVA FUNÇÃO: Debug completo de sequência de calibração
        Com opção de corrigir automaticamente os problemas encontrados
        """
        if not self._check_connection():
            return {"error": "Robô não conectado"}
        
        self.logger.info(f" DEBUG de sequência de calibração - {len(poses)} poses")
        
        debug_results = {
            "sequence_analysis": {
                "total_poses": len(poses),
                "valid_poses": 0,
                "invalid_poses": 0,
                "correctable_poses": 0,
                "impossible_poses": 0
            },
            "pose_details": [],
            "problems_found": [],
            "solutions_applied": [],
            "final_sequence": None,
            "recommendations": []
        }
        
        corrected_sequence = []
        
        for i, pose in enumerate(poses):
            self.logger.info(f"📍 Analisando pose {i+1}: {pose}")
            
            pose_debug = {
                "index": i,
                "original_pose": asdict(pose),
                "issues": [],
                "corrections": [],
                "final_status": "unknown"
            }
            
            try:
                # 1. Debug detalhado da pose
                detailed_debug = self.debug_pose_detailed(pose)
                
                # 2. Verificar se pose é válida
                pose_list = pose.to_list()
                is_valid = self.controller.validate_pose_complete(pose_list)
                
                if is_valid:
                    debug_results["sequence_analysis"]["valid_poses"] += 1
                    pose_debug["final_status"] = " VÁLIDA"
                    corrected_sequence.append(pose)
                else:
                    debug_results["sequence_analysis"]["invalid_poses"] += 1
                    
                    # 3. Identificar problemas específicos
                    diagnostics = detailed_debug.get("pose_analysis", {})
                    
                    if not diagnostics.get("pose_alcancavel", True):
                        pose_debug["issues"].append("Cinemática inversa impossível")
                    
                    if diagnostics.get("conflitos_base_ferro", False):
                        pose_debug["issues"].append("Conflito com base de ferro")
                    
                    if diagnostics.get("joints_problematicas", []):
                        pose_debug["issues"].append("Articulações fora dos limites")
                    
                    if diagnostics.get("singularidades", False):
                        pose_debug["issues"].append("Próximo a singularidades")
                    
                    # 4. Tentar correção se solicitado
                    if fix_problems:
                        self.logger.info(f"🔧 Tentando corrigir pose {i+1}...")
                        
                        corrected_pose_list, success = self.controller.fix_calibration_pose(i, pose_list)
                        
                        if success:
                            corrected_pose = RobotPose.from_list(corrected_pose_list)
                            pose_debug["corrections"].append("Pose corrigida automaticamente")
                            pose_debug["corrected_pose"] = asdict(corrected_pose)
                            pose_debug["final_status"] = "🔧 CORRIGIDA"
                            debug_results["sequence_analysis"]["correctable_poses"] += 1
                            debug_results["solutions_applied"].append(f"Pose {i+1} corrigida")
                            corrected_sequence.append(corrected_pose)
                        else:
                            pose_debug["final_status"] = " IMPOSSÍVEL"
                            debug_results["sequence_analysis"]["impossible_poses"] += 1
                            debug_results["problems_found"].append(f"Pose {i+1} não pode ser corrigida")
                            
                            # Adicionar pose segura como fallback
                            safe_pose = self._generate_safe_fallback_pose(pose)
                            corrected_sequence.append(safe_pose)
                            pose_debug["corrections"].append("Substituída por pose segura")
                    else:
                        pose_debug["final_status"] = " INVÁLIDA (não corrigida)"
                    
            except Exception as e:
                pose_debug["final_status"] = f" ERRO: {str(e)}"
                pose_debug["issues"].append(f"Erro na análise: {str(e)}")
                
            debug_results["pose_details"].append(pose_debug)
        
        # Gerar sequência final se correções foram aplicadas
        if fix_problems:
            debug_results["final_sequence"] = [asdict(pose) for pose in corrected_sequence]
        
        # Gerar recomendações
        invalid_rate = (debug_results["sequence_analysis"]["invalid_poses"] / len(poses)) * 100
        
        if invalid_rate > 30:
            debug_results["recommendations"].append("Taxa alta de poses inválidas - revisar configuração do workspace")
        
        if debug_results["sequence_analysis"]["impossible_poses"] > 0:
            debug_results["recommendations"].append("Poses impossíveis detectadas - verificar limites físicos do robô")
        
        if any("base de ferro" in str(detail.get("issues", [])) for detail in debug_results["pose_details"]):
            debug_results["recommendations"].append("Conflitos com base de ferro - considerar elevar altura das poses")
        
        # Log resumo
        self.logger.info(" RESUMO DO DEBUG:")
        self.logger.info(f"   Válidas: {debug_results['sequence_analysis']['valid_poses']}")
        self.logger.info(f"   Inválidas: {debug_results['sequence_analysis']['invalid_poses']}")
        if fix_problems:
            self.logger.info(f"   Corrigidas: {debug_results['sequence_analysis']['correctable_poses']}")
            self.logger.info(f"   Impossíveis: {debug_results['sequence_analysis']['impossible_poses']}")
        
        return debug_results

    def _generate_safe_fallback_pose(self, problematic_pose: RobotPose) -> RobotPose:
        """
        🛡️ FUNÇÃO AUXILIAR: Gera pose segura como fallback
        """
        # Usar configuração da base de ferro se disponível
        iron_config = self.config.get("iron_base_constraint", {})
        safe_z = iron_config.get("base_height", 0.1) + iron_config.get("safety_margin", 0.05) + 0.1
        
        # Criar pose segura mantendo X,Y mas elevando Z
        safe_pose = RobotPose(
            x=max(-0.5, min(0.5, problematic_pose.x)),  # Limitar X
            y=max(-0.3, min(0.3, problematic_pose.y)),  # Limitar Y  
            z=max(safe_z, 0.3),  # Z seguro
            rx=0.0,  # Orientação conservadora
            ry=3.14,  # TCP para baixo
            rz=0.0
        )
        
        self.logger.info(f" Pose segura gerada: {safe_pose}")
        return safe_pose

    def benchmark_correction_system(self) -> Dict[str, Any]:
        """
        📊 NOVA FUNÇÃO: Benchmark do sistema de correção
        Testa o sistema com poses conhecidas
        """
        if not self._check_connection():
            return {"error": "Robô não conectado"}
        
        self.logger.info(" Iniciando benchmark do sistema de correção")
        
        # Usar função do URController
        benchmark_results = self.controller.benchmark_correction_system()
        
        # Adicionar análise do RobotService
        service_analysis = {
            "controller_results": benchmark_results,
            "service_config": {
                "validation_level": self.config.get("default_validation_level"),
                "correction_enabled": self.config.get("enable_auto_correction"),
                "iron_base_enabled": self.config.get("iron_base_constraint", {}).get("enabled", False)
            },
            "performance_rating": "unknown",
            "recommendations": []
        }
        
        # Calcular rating de performance
        if benchmark_results.get("total", 0) > 0:
            correction_rate = (benchmark_results.get("corrected_valid", 0) - 
                            benchmark_results.get("original_valid", 0)) / benchmark_results.get("total", 1) * 100
            
            if correction_rate > 50:
                service_analysis["performance_rating"] = " EXCELENTE"
            elif correction_rate > 20:
                service_analysis["performance_rating"] = " BOM"
            elif correction_rate > 0:
                service_analysis["performance_rating"] = " REGULAR"
            else:
                service_analysis["performance_rating"] = " RUIM"
            
            # Recomendações baseadas no desempenho
            if correction_rate < 10:
                service_analysis["recommendations"].append("Sistema de correção pouco efetivo - verificar configurações")
            
            impossible_rate = benchmark_results.get("impossible", 0) / benchmark_results.get("total", 1) * 100
            if impossible_rate > 30:
                service_analysis["recommendations"].append("Muitas poses impossíveis - revisar workspace limits")
        
        self.logger.info(f" Benchmark concluído - Rating: {service_analysis['performance_rating']}")
        
        return service_analysis

    # =================== CONTEXT MANAGER ===================

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()

# ===================  CLASSE DE UTILIDADE PARA CRIAÇÃO DE COMANDOS ===================

class MovementCommandBuilder:
    """
     NOVA CLASSE: Builder pattern para criar comandos de movimento facilmente
    """
    @staticmethod
    def create_linear_movement(pose: RobotPose, 
                             speed: float = None,
                             acceleration: float = None,
                             validation_level: ValidationLevel = ValidationLevel.ADVANCED,
                             movement_strategy: MovementStrategy = MovementStrategy.SMART_CORRECTION) -> MovementCommand:
        """Cria comando de movimento linear"""
        return MovementCommand(
            type=MovementType.LINEAR,
            target_pose=pose,
            speed=speed,
            acceleration=acceleration,
            validation_level=validation_level,
            movement_strategy=movement_strategy
        )
    
    @staticmethod
    def create_home_movement(validation_level: ValidationLevel = ValidationLevel.COMPLETE) -> MovementCommand:
        """Cria comando de movimento para home"""
        return MovementCommand(
            type=MovementType.HOME,
            validation_level=validation_level,
            movement_strategy=MovementStrategy.ULTRA_SAFE
        )
    
    @staticmethod
    def create_pick_place_movement(origin: RobotPose,
                                 destination: RobotPose,
                                 safe_height: float = 0.3,
                                 pick_height: float = 0.05) -> MovementCommand:
        """Cria comando de pick and place"""
        return MovementCommand(
            type=MovementType.PICK_PLACE,
            validation_level=ValidationLevel.COMPLETE,
            movement_strategy=MovementStrategy.ULTRA_SAFE,
            parameters={
                "origin": origin.to_list(),
                "destination": destination.to_list(),
                "safe_height": safe_height,
                "pick_height": pick_height
            }
        )
    
    @staticmethod
    def create_sequence_from_poses(poses: List[RobotPose],
                                 speed: float = None,
                                 validation_level: ValidationLevel = ValidationLevel.ADVANCED,
                                 movement_strategy: MovementStrategy = MovementStrategy.SMART_CORRECTION) -> List[MovementCommand]:
        """Cria sequência de comandos a partir de lista de poses"""
        return [
            MovementCommandBuilder.create_linear_movement(
                pose, speed, None, validation_level, movement_strategy
            ) for pose in poses
        ]