import json
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

from logic_control.ur_controller import URController

class RobotStatus(Enum):
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    MOVING = "moving"
    IDLE = "idle"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"

class MovementType(Enum):
    LINEAR = "linear"
    JOINT = "joint"
    PICK_PLACE = "pick_place"
    HOME = "home"

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
    parameters: Optional[Dict[str, Any]] = None

@dataclass
class PickPlaceCommand:
    origin: RobotPose
    destination: RobotPose
    safe_height: float
    pick_height: float
    speed_normal: float = 0.1
    speed_precise: float = 0.05

class RobotService:
    def __init__(self, robot_ip: str = "10.1.5.37", config_file: Optional[str] = None):
        self.robot_ip = robot_ip
        self.controller: Optional[URController] = None
        self.status = RobotStatus.CONNECTED
        self.last_error: Optional[str] = None
        
        # Configurações padrão
        self.default_config = {
            "speed": 0.1,
            "acceleration": 0.1,
            "safe_height": 0.3,
            "pick_height": 0.05,
            "pause_between_moves": 1.0,
            "home_pose": [0.4, 0.0, 0.4, 0.0, 3.14, 0.0],
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
            self.logger.info(f"Conectando ao robô em {self.robot_ip}...")
            self.controller = URController(
                robot_ip=self.robot_ip,
                speed=self.config["speed"],
                acceleration=self.config["acceleration"]
            )
            
            if self.controller.is_connected():
                self.status = RobotStatus.CONNECTED
                self.logger.info(" Robô conectado com sucesso")
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
            self.logger.info("🔌 Robô desconectado")
        except Exception as e:
            self.logger.error(f" Erro ao desconectar: {e}")

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
        """Retorna status completo do serviço"""
        status_dict = {
            "status": self.status.value,
            "connected": self.status not in [RobotStatus.DISCONNECTED, RobotStatus.ERROR],
            "last_error": self.last_error,
            "current_pose": None,
            "robot_details": None
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

    def move_to_pose(self, pose: RobotPose, speed: Optional[float] = None, acceleration: Optional[float] = None) -> bool:
        """Move robô para pose específica"""
        if not self._check_connection():
            return False
            
        try:
            self.status = RobotStatus.MOVING
            self.logger.info(f" Movendo para: {pose}")
            
            # Usar velocidades especificadas ou padrão
            move_speed = speed or self.config["speed"]
            move_acceleration = acceleration or self.config["acceleration"]
            
            # Atualizar parâmetros do controlador se necessário
            if speed or acceleration:
                self.controller.set_speed_parameters(move_speed, move_acceleration)
            
            success = self.controller.move_to_pose_safe(pose.to_list())
            
            if success:
                self.status = RobotStatus.IDLE
                self.logger.info(" Movimento concluído")
                return True
            else:
                self.status = RobotStatus.ERROR
                self.last_error = "Falha no movimento"
                self.logger.error(" Falha no movimento")
                return False
                
        except Exception as e:
            self.status = RobotStatus.ERROR
            self.last_error = str(e)
            self.logger.error(f" Erro durante movimento: {e}")
            return False

    def move_home(self) -> bool:
        """Move robô para posição home"""
        home_pose = RobotPose.from_list(self.config["home_pose"])
        self.logger.info(" Movendo para posição home")
        return self.move_to_pose(home_pose)

    def pick_and_place(self, pick_place_cmd: PickPlaceCommand) -> bool:
        """Executa operação de pegar e colocar"""
        if not self._check_connection():
            return False
            
        try:
            self.status = RobotStatus.MOVING
            self.logger.info(f" Iniciando pick and place:")
            self.logger.info(f"    Origem: {pick_place_cmd.origin}")
            self.logger.info(f"    Destino: {pick_place_cmd.destination}")
            
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

    def execute_sequence(self, commands: List[MovementCommand]) -> bool:
        """Executa sequência de comandos"""
        if not self._check_connection():
            return False
            
        self.logger.info(f" Executando sequência de {len(commands)} comandos")
        
        for i, cmd in enumerate(commands):
            self.logger.info(f"⏯Executando comando {i+1}/{len(commands)}: {cmd.type.value}")
            
            if cmd.type == MovementType.LINEAR:
                if not self.move_to_pose(cmd.target_pose, cmd.speed, cmd.acceleration):
                    self.logger.error(f" Falha no comando {i+1}")
                    return False
                    
            elif cmd.type == MovementType.HOME:
                if not self.move_home():
                    self.logger.error(f" Falha no comando home {i+1}")
                    return False
                    
            elif cmd.type == MovementType.PICK_PLACE:
                if cmd.parameters:
                    pick_place_cmd = PickPlaceCommand(
                        origin=RobotPose.from_list(cmd.parameters["origin"]),
                        destination=RobotPose.from_list(cmd.parameters["destination"]),
                        safe_height=cmd.parameters.get("safe_height", self.config["safe_height"]),
                        pick_height=cmd.parameters.get("pick_height", self.config["pick_height"])
                    )
                    if not self.pick_and_place(pick_place_cmd):
                        self.logger.error(f" Falha no comando pick_place {i+1}")
                        return False
            
            # Pausa entre comandos
            time.sleep(self.config["pause_between_moves"])
        
        self.logger.info(" Sequência executada com sucesso")
        return True

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
        elif pose_name in self.config["workspace_poses"]:
            return RobotPose.from_list(self.config["workspace_poses"][pose_name])
        else:
            self.logger.error(f" Pose '{pose_name}' não encontrada")
            return None

    def update_config(self, new_config: Dict[str, Any]):
        """Atualiza configuração do serviço"""
        self.config.update(new_config)
        self.logger.info(" Configuração atualizada")
        
        # Atualizar parâmetros do controlador se conectado
        if self.controller and "speed" in new_config or "acceleration" in new_config:
            self.controller.set_speed_parameters(
                self.config["speed"],
                self.config["acceleration"]
            )

    def _check_connection(self) -> bool:
        """Verifica se está conectado ao robô"""
        if not self.controller or not self.controller.is_connected():
            self.status = RobotStatus.DISCONNECTED
            self.last_error = "Robô não conectado"
            self.logger.error(" Robô não está conectado")
            return False
        return True

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()