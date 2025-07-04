import ur_rtde.rtde as rtde
import ur_rtde.rtde_config as rtde_config
import numpy as np


class URController:
    """
    Classe para controle do robô UR via interface RTDE
    
    Atributos:
        rtde_c (rtde.RTDE): Conexão com o robô
        config (rtde_config.ConfigFile): Configuração da interface
    """
    
    def __init__(self, robot_ip: str, config_file: str = 'rtde_config.xml'):
        """
        Inicializa conexão com o robô UR
        
        Args:
            robot_ip (str): Endereço IP do robô
            config_file (str): Arquivo de configuração RTDE
        """
        self.rtde_c = rtde.RTDE(robot_ip, 30004)
        self.config = rtde_config.ConfigFile(config_file)
        self.setup_connection()
    
    def setup_connection(self):
        """Configura conexão RTDE com o robô"""
        self.rtde_c.connect()
        
        if not self.rtde_c.is_connected():
            raise ConnectionError("Falha ao conectar ao robô UR")
        
        # Configurar recipes para leitura/escrita
        self.state_names, self.state_types = self.config.get_recipe('state')
        self.setp_names, self.setp_types = self.config.get_recipe('setp')
        self.watchdog_names, self.watchdog_types = self.config.get_recipe('watchdog')
        
        # Configurar controle
        self.rtde_c.send_output_setup(self.state_names, self.state_types)
        self.rtde_c.send_input_setup(self.setp_names, self.setp_types)
    
    def get_current_pose(self) -> np.ndarray:
        """Obtém a pose atual do robô"""
        return self.rtde_c.receive().actual_TCP_pose
    
    def move_to_pose(self, pose: np.ndarray, 
                   velocity: float = 0.2, 
                   acceleration: float = 0.5) -> bool:
        """
        Move o robô para uma pose específica
        
        Args:
            pose (np.ndarray): Pose alvo [x, y, z, rx, ry, rz]
            velocity (float): Velocidade normalizada (0-1)
            acceleration (float): Aceleração normalizada (0-1)
            
        Returns:
            bool: True se movimento foi iniciado com sucesso
        """
        # Verificar se pose é válida
        if len(pose) != 6:
            raise ValueError("Pose deve conter 6 valores [x,y,z,rx,ry,rz]")
        
        # Configurar movimento
        self.rtde_c.send(self.setp_names, self.setp_types, pose)
        return True
    
    def generate_pick_script(self, 
                           approach_pose: np.ndarray, 
                           target_pose: np.ndarray, 
                           retreat_pose: np.ndarray) -> str:
        """
        Gera script URScript para sequência de pick-and-place
        
        Args:
            approach_pose (np.ndarray): Pose de aproximação
            target_pose (np.ndarray): Pose para pegar objeto
            retreat_pose (np.ndarray): Pose de retirada
            
        Returns:
            str: Script URScript completo
        """
        script = f"""
def pick_sequence():
    # Abrir gripper
    set_tool_digital_out(0, False)
    sleep(0.5)
    
    # Movimento de aproximação
    movel(p{approach_pose.tolist()}, a={0.5}, v={0.2})
    
    # Movimento para pegar
    movel(p{target_pose.tolist()}, a={0.3}, v={0.1})
    
    # Fechar gripper
    set_tool_digital_out(0, True)
    sleep(0.5)
    
    # Movimento de retirada
    movel(p{retreat_pose.tolist()}, a={0.5}, v={0.2})
end
"""
        return script
    
    def execute_script(self, script: str):
        """Executa script URScript no robô"""
        self.rtde_c.send_program(script)
    
    def disconnect(self):
        """Desconecta do robô"""
        self.rtde_c.disconnect()