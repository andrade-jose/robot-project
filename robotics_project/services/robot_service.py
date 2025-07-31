import json
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path
import math

from logic_control.ur_controller import URController

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
    def __init__(self, robot_ip: str = "10.1.5.92", config_file: Optional[str] = None):
        self.robot_ip = robot_ip
        self.controller: Optional[URController] = None
        self.status = RobotStatus.DISCONNECTED
        self.last_error: Optional[str] = None
        
        #  CONFIGURAÇÕES ATUALIZADAS com novos parâmetros de segurança
        self.default_config = {
            "speed": 0.1,
            "acceleration": 0.1,
            "safe_height": 0.3,
            "pick_height": 0.05,
            "pause_between_moves": 1.0,
            "home_pose": [0.4, 0.0, 0.4, 0.0, 3.14, 0.0],
            
            #  NOVAS CONFIGURAÇÕES DE SEGURANÇA
            "default_validation_level": "advanced",
            "default_movement_strategy": "smart_correction",
            "enable_auto_correction": True,
            "max_correction_attempts": 3,
            "intermediate_points_distance_threshold": 0.3,  # Usa pontos intermediários se movimento > 30cm
            "ultra_safe_mode": False,  # Modo ultra-seguro (muito lento mas máxima segurança)
            
            # Configurações de movimento inteligente
            "smart_movement": {
                "enable_smart_correction": True,
                "enable_intermediate_points": True,
                "max_movement_distance": 1.0,
                "intermediate_points_step": 0.2,  # 20cm entre pontos
                "ultra_safe_speed_factor": 0.3,
                "validation_retries": 3
            },

            "iron_base_constraint": {
                "enabled": True,
                "base_height": 0.05,  # Altura da base de ferro
                "safety_margin": 0.02,
                "shoulder_offset": 0.3  # Offset estimado do ombro ao TCP
            },
            
            "workspace_poses": {
                "center": [0.3, 0.0, 0.2, 0.0, 3.14, 0.0],
                "left": [0.3, 0.3, 0.2, 0.0, 3.14, 0.0],
                "right": [0.3, -0.3, 0.2, 0.0, 3.14, 0.0],
                "front": [0.5, 0.0, 0.2, 0.0, 3.14, 0.0],
                "back": [0.1, 0.0, 0.2, 0.0, 3.14, 0.0]
            }

        }
        
        # Carregar configuração se fornecida
        self.config = self.load_config(config_file) if config_file else self.default_config.copy()
        
        # Setup logging
        self.setup_logging()
        
        #  NOVO: Histórico de movimentos para análise
        self.movement_history: List[Dict] = []
        self.validation_stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "corrections_applied": 0,
            "movements_with_intermediate_points": 0
        }
        self.verbose_logging = False  # Controla logs detalhados
        self.log_summary_only = True  # Apenas logs de resumo

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

    def load_config(self, config_file: str) -> Dict:
        """Carrega configuração de arquivo JSON"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            self.logger.info(f" Configuração carregada de {config_file}")
            return {**self.default_config, **config}
        except Exception as e:
            self.logger.error(f" Erro ao carregar configuração: {e}")
            return self.default_config.copy()

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
            
        # Estimar posição do ombro
        estimated_shoulder_z = pose.z - self.config["iron_base_constraint"]["shoulder_offset"]
        min_height = (self.config["iron_base_constraint"]["base_height"] + 
                    self.config["iron_base_constraint"]["safety_margin"])
        
        if estimated_shoulder_z < min_height:
            if self.verbose_logging:
                self.logger.warning(f"⚠️ Pose rejeitada - ombro muito baixo: {estimated_shoulder_z:.3f}m")
            return False
        
        return True

    def find_alternative_pose(self, problematic_pose: RobotPose) -> Optional[RobotPose]:
        """
        🔥 NOVA: Encontra pose alternativa quando original é inviável
        """
        if not self.validate_iron_base_constraint(problematic_pose):
            # Elevar TCP para proteger ombro
            min_tcp_z = (self.config["iron_base_constraint"]["base_height"] + 
                        self.config["iron_base_constraint"]["safety_margin"] + 
                        self.config["iron_base_constraint"]["shoulder_offset"] + 0.05)  # +5cm segurança
            
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
                    self.logger.info(f"Pose corrigida para proteger ombro: Z {problematic_pose.z:.3f} -> {min_tcp_z:.3f}")
                
                return alternative
        
        return None
    # ===================  FUNÇÕES DE MOVIMENTO ATUALIZADAS ===================

    def move_to_pose(self, pose: RobotPose, speed: Optional[float] = None, 
                    acceleration: Optional[float] = None,
                    movement_strategy: MovementStrategy = None,
                    validation_level: ValidationLevel = None) -> bool:
        """
         FUNÇÃO ATUALIZADA: Move robô com estratégias inteligentes de segurança
        """
        if not self._check_connection():
            return False
        
        if not self.validate_iron_base_constraint(pose):
            alternative_pose = self.find_alternative_pose(pose)
            if alternative_pose:
                pose = alternative_pose  # Usar pose corrigida
            else:
                self.status = RobotStatus.ERROR
                self.last_error = "Pose inviável - ombro abaixo da base"
                self.logger.error("❌ Movimento impossível - limitação da base de ferro")
                return False
        
        # Usar configurações padrão se não especificadas
        if movement_strategy is None:
            movement_strategy = MovementStrategy(self.config.get("default_movement_strategy", "smart_correction"))
        
        if validation_level is None:
            validation_level = ValidationLevel(self.config.get("default_validation_level", "advanced"))
            
        try:
            self.status = RobotStatus.MOVING
            if self.verbose_logging:
                self.logger.info(f" Movendo para: {pose}")
                self.logger.info(f" Estratégia: {movement_strategy.value}, Validação: {validation_level.value}")
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
                "speed": move_speed,
                "acceleration": move_acceleration
            }
            
            success = False
            pose_list = pose.to_list()
            
            #  EXECUTAR ESTRATÉGIA DE MOVIMENTO
            if movement_strategy == MovementStrategy.DIRECT:
                # Movimento direto (modo legado)
                success = self.controller.move_to_pose_safe(pose_list, move_speed, move_acceleration, use_smart_correction=False)
                
            elif movement_strategy == MovementStrategy.SMART_CORRECTION:
                #  NOVO: Movimento com correção automática
                success, corrected_pose = self.controller.move_to_pose_with_smart_correction(
                    pose_list, move_speed, move_acceleration, 
                    max_correction_attempts=self.config.get("max_correction_attempts", 3)
                )
                if corrected_pose:
                    movement_record["corrected_pose"] = corrected_pose
                    self.validation_stats["corrections_applied"] += 1
                    
            elif movement_strategy == MovementStrategy.INTERMEDIATE:
                #  NOVO: Movimento com pontos intermediários
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
                        
            elif movement_strategy == MovementStrategy.ULTRA_SAFE:
                #  NOVO: Estratégia ultra-segura (todas as validações e correções)
                success = self.controller.move_to_pose_safe(
                    pose_list, move_speed, move_acceleration, use_smart_correction=True
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
                    self.logger.info(f"Movimento concluído - {movement_strategy.value}")
                return True
            else:
                self.status = RobotStatus.ERROR
                self.last_error = "Falha no movimento"
                self.logger.error(f" Movimento falhou - {movement_strategy.value}")
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
                        self.logger.info(f"🔧 Ponto intermediário {i} corrigido")
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
                    self.logger.warning("🚨 PARADA DE EMERGÊNCIA ATIVADA")
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
                    self.logger.info("🛑 Movimento parado")
                    return True
            return False
        except Exception as e:
            self.logger.error(f" Erro ao parar movimento: {e}")
            return False

    def get_predefined_pose(self, pose_name: str) -> Optional[RobotPose]:
        """Obtém pose predefinida por nome"""
        if pose_name == "home":
            return RobotPose.from_list(self.config["home_pose"])
        elif pose_name in self.config["workspace_poses"]:
            return RobotPose.from_list(self.config["workspace_poses"][pose_name])
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
            
            self.logger.info(f"📄 Histórico exportado para: {filename}")
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

    def configure_iron_base(self, height: float, margin: float = 0.02, shoulder_offset: float = 0.3):
        """
        🔥 NOVA: Configura parâmetros da base de ferro
        """
        self.config["iron_base_constraint"].update({
            "base_height": height,
            "safety_margin": margin,
            "shoulder_offset": shoulder_offset
        })
        
        # Também configurar no URController
        if self.controller:
            self.controller.set_iron_base_height(height)
        
        self.logger.info(f"🔧 Base de ferro configurada: {height:.3f}m + {margin:.3f}m margem")

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