import ur_rtde.rtde as rtde
import ur_rtde.rtde_config as rtde_config
import time

class URController:
    def __init__(self, robot_ip, config_file='config/rtde_config.xml'):
        self.rtde_c = rtde.RTDE(robot_ip, 30004)
        self.config = rtde_config.ConfigFile(config_file)
        self.state_names, self.state_types = self.config.get_recipe('state')
        self.setp_names, self.setp_types = self.config.get_recipe('setp')
        self.rtde_c.connect()
        self.rtde_c.send_output_setup(self.state_names, self.state_types)
        self.rtde_c.send_input_setup(self.setp_names, self.setp_types)
    
    def move_to(self, position):
        pose = list(position) + [0, 0, 0]  # Posição + orientação
        self.rtde_c.send(self.setp_names, self.setp_types, pose)
        time.sleep(1)
        
    def disconnect(self):
        self.rtde_c.disconnect()