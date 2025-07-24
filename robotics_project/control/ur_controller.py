from rtde_control import RTDEControlInterface

class URController:
    def __init__(self, robot_ip="10.1.6.119", speed=0.25, acceleration=0.5):
        self.robot_ip = robot_ip
        self.speed = speed
        self.acceleration = acceleration
        self.rtde_c = RTDEControlInterface(self.robot_ip)

    def is_connected(self):
        return self.rtde_c and self.rtde_c.isConnected()

    def move_to_pose(self, pose, speed=None, acceleration=None):
        """
        Move para uma pose absoluta (lista de 6 floats: x,y,z,rx,ry,rz)
        """
        if not self.is_connected():
            raise ConnectionError("Robô não está conectado.")

        spd = speed if speed else self.speed
        acc = acceleration if acceleration else self.acceleration
        self.rtde_c.moveL(pose, spd, acc)

    def stop(self):
        if self.rtde_c:
            self.rtde_c.stopScript()

    def disconnect(self):
        if self.rtde_c:
            self.rtde_c.disconnect()

    def move_home(self, home_pose):
        self.move_to_pose(home_pose)
