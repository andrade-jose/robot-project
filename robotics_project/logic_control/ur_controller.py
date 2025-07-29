from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
import time
import math

class URController:
    def __init__(self, robot_ip="10.1.5.37", speed=0.1, acceleration=0.1):
        self.robot_ip = robot_ip
        self.speed = speed
        self.acceleration = acceleration
    
        self.rtde_c = RTDEControlInterface(self.robot_ip)
        self.rtde_r = RTDEReceiveInterface(self.robot_ip)
        print(f"✅ Conectado ao robô UR em {self.robot_ip}")


        # Parâmetros de segurança
        self.pause_between_moves = 1.0  # Aumentado para maior segurança
        self.safe_height_offset = 0.1   # Altura segura acima do tabuleiro
        self.min_elbow_tcp_distance = 0.028
        self.last_error = None
        self.em_movimento = False

        # Configurações de movimento mais conservadoras
        self.max_joint_change = 0.3  # Reduzido para movimentos mais suaves
        self.planning_steps = 10
        
        # Limites de workspace (ajuste conforme seu robô)
        self.workspace_limits = {
            'x_min': -0.8, 'x_max': 0.8,
            'y_min': -0.8, 'y_max': 0.8,
            'z_min': 0.05, 'z_max': 0.8,  # Z mínimo para evitar colisão com base
            'rx_min': -math.pi, 'rx_max': math.pi,
            'ry_min': -math.pi, 'ry_max': math.pi,
            'rz_min': -math.pi, 'rz_max': math.pi
        }

    def is_connected(self):
        """Verifica se está conectado ao robô"""
        return (self.rtde_c and 
                self.rtde_r and 
                self.rtde_c.isConnected())

    def validate_pose(self, pose):
        """
        Valida se a pose está dentro dos limites do workspace
        Pose format: [x, y, z, rx, ry, rz] onde:
        - x, y, z em metros
        - rx, ry, rz em radianos (angle-axis representation)
        """
        if len(pose) != 6:
            print(f"❌ Pose inválida: deve ter 6 elementos, recebeu {len(pose)}")
            return False
            
        x, y, z, rx, ry, rz = pose
        
        # Validar posição cartesiana
        if not (self.workspace_limits['x_min'] <= x <= self.workspace_limits['x_max']):
            print(f"❌ X fora dos limites: {x} (min: {self.workspace_limits['x_min']}, max: {self.workspace_limits['x_max']})")
            return False
            
        if not (self.workspace_limits['y_min'] <= y <= self.workspace_limits['y_max']):
            print(f"❌ Y fora dos limites: {y} (min: {self.workspace_limits['y_min']}, max: {self.workspace_limits['y_max']})")
            return False
            
        if not (self.workspace_limits['z_min'] <= z <= self.workspace_limits['z_max']):
            print(f"❌ Z fora dos limites: {z} (min: {self.workspace_limits['z_min']}, max: {self.workspace_limits['z_max']})")
            return False
        
        # Validar orientação (angle-axis)
        rotation_magnitude = math.sqrt(rx**2 + ry**2 + rz**2)
        if rotation_magnitude > math.pi:
            print(f"❌ Magnitude de rotação muito grande: {rotation_magnitude} > π")
            return False
            
        print(f"✅ Pose válida: x={x:.3f}, y={y:.3f}, z={z:.3f}, rx={rx:.3f}, ry={ry:.3f}, rz={rz:.3f}")
        return True

    def get_current_pose(self):
        """Retorna a pose atual do TCP"""
        if self.is_connected():
            try:
                pose = self.rtde_r.getActualTCPPose()
                if pose:
                    print(f"📍 Pose atual: x={pose[0]:.3f}, y={pose[1]:.3f}, z={pose[2]:.3f}, "
                          f"rx={pose[3]:.3f}, ry={pose[4]:.3f}, rz={pose[5]:.3f}")
                return pose
            except Exception as e:
                print(f"❌ Erro ao obter pose atual: {e}")
                return None
        return None

    def getActualTCPPose(self):
        """Alias para compatibilidade"""
        return self.get_current_pose()

    def get_current_joints(self):
        """Retorna as posições atuais das juntas"""
        if self.is_connected():
            try:
                joints = self.rtde_r.getActualQ()
                if joints:
                    print(f"🔧 Juntas atuais: {[f'{j:.3f}' for j in joints]}")
                return joints
            except Exception as e:
                print(f"❌ Erro ao obter juntas: {e}")
                return None
        return None

    def is_pose_reachable(self, target_pose):
        """
        Verifica se a pose é alcançável fazendo uma validação básica
        """
        current_pose = self.get_current_pose()
        if not current_pose:
            return False
            
        # Calcular distância euclidiana
        distance = math.sqrt(
            (target_pose[0] - current_pose[0])**2 +
            (target_pose[1] - current_pose[1])**2 +
            (target_pose[2] - current_pose[2])**2
        )
        
        # Verificar se a distância não é muito grande
        max_distance = 1.0  # 1 metro máximo
        if distance > max_distance:
            print(f"❌ Distância muito grande: {distance:.3f}m > {max_distance}m")
            return False
            
        return True

    def move_to_pose_safe(self, pose, speed=None, acceleration=None):
        """
        Movimento seguro para uma pose com validações
        """
        if speed is None:
            speed = self.speed
        if acceleration is None:
            acceleration = self.acceleration
            
        print(f"🎯 Iniciando movimento para pose: {[f'{p:.3f}' for p in pose]}")
        
        if not self.is_connected():
            print("❌ Robô não está conectado")
            return False
            
        # Validar pose
        if not self.validate_pose(pose):
            print("❌ Pose inválida")
            return False
            
        # Verificar se é alcançável
        if not self.is_pose_reachable(pose):
            print("❌ Pose não alcançável")
            return False
            
        try:
            self.em_movimento = True
            
            # Executar movimento
            success = self.rtde_c.moveL(pose, speed, acceleration)
            
            if success:
                # Aguardar conclusão do movimento
                time.sleep(self.pause_between_moves)
                print("✅ Movimento concluído com sucesso")
                return True
            else:
                print("❌ Falha no comando moveL")
                return False
                
        except Exception as e:
            print(f"❌ Erro durante movimento: {e}")
            return False
        finally:
            self.em_movimento = False

    def move_to_pose(self, pose):
        """Interface compatível com código existente"""
        return self.move_to_pose_safe(pose)

    def move_home(self, pose_home):
        """Move para posição home"""
        print(f"🏠 Movendo para posição home")
        return self.move_to_pose_safe(pose_home)

    def executar_movimento_peca(self, origem, destino, altura_segura, altura_pegar):
        """
        Executa sequência de movimentos para pegar e colocar peça
        """
        print(f"🤖 Executando movimento de peça:")
        print(f"   📍 Origem: {[f'{p:.3f}' for p in origem]}")
        print(f"   📍 Destino: {[f'{p:.3f}' for p in destino]}")
        print(f"   ⬆️ Altura segura: {altura_segura:.3f}")
        print(f"   ⬇️ Altura pegar: {altura_pegar:.3f}")
        
        try:
            # 1. Mover para posição segura acima da origem
            pose_segura_origem = origem.copy()
            pose_segura_origem[2] = altura_segura
            
            if not self.move_to_pose_safe(pose_segura_origem):
                print("❌ Falha ao mover para posição segura origem")
                return False
                
            # 2. Descer para pegar a peça
            pose_pegar = origem.copy()
            pose_pegar[2] = altura_pegar
            
            if not self.move_to_pose_safe(pose_pegar, speed=0.05):  # Movimento mais lento
                print("❌ Falha ao descer para pegar peça")
                return False
                
            # 3. Subir com a peça
            if not self.move_to_pose_safe(pose_segura_origem):
                print("❌ Falha ao subir com peça")
                return False
                
            # 4. Mover para posição segura acima do destino
            pose_segura_destino = destino.copy()
            pose_segura_destino[2] = altura_segura
            
            if not self.move_to_pose_safe(pose_segura_destino):
                print("❌ Falha ao mover para posição segura destino")
                return False
                
            # 5. Descer para colocar a peça
            pose_colocar = destino.copy()
            pose_colocar[2] = altura_pegar
            
            if not self.move_to_pose_safe(pose_colocar, speed=0.05):  # Movimento mais lento
                print("❌ Falha ao descer para colocar peça")
                return False
                
            # 6. Subir após colocar
            if not self.move_to_pose_safe(pose_segura_destino):
                print("❌ Falha ao subir após colocar peça")
                return False
                
            print("✅ Movimento de peça concluído com sucesso")
            return True
            
        except Exception as e:
            print(f"❌ Erro durante movimento de peça: {e}")
            return False

    def emergency_stop(self):
        """Parada de emergência"""
        try:
            if self.rtde_c:
                self.rtde_c.stopScript()
                self.em_movimento = False
                print("🚨 PARADA DE EMERGÊNCIA ATIVADA!")
                return True
        except Exception as e:
            print(f"❌ Erro na parada de emergência: {e}")
            return False

    def stop(self):
        """Para movimentos atuais"""
        try:
            if self.rtde_c and self.em_movimento:
                self.rtde_c.stopL(2.0)  # Para movimento linear com desaceleração
                self.em_movimento = False
                print("🛑 Robô parado com sucesso")
                return True
            return True
        except Exception as e:
            print(f"❌ Erro ao parar robô: {e}")
            return False

    def set_speed_parameters(self, speed, acceleration):
        """Ajusta parâmetros de velocidade"""
        # Limites de segurança mais conservadores
        self.speed = max(0.005, min(speed, 0.2))  # 5mm/s a 200mm/s
        self.acceleration = max(0.005, min(acceleration, 0.2))
        print(f"⚙️ Parâmetros atualizados - Velocidade: {self.speed:.3f}, Aceleração: {self.acceleration:.3f}")

    def get_robot_status(self):
        """Obtém status detalhado do robô"""
        if not self.is_connected():
            return {"connected": False}
            
        try:
            status = {
                "connected": True,
                "pose": self.get_current_pose(),
                "joints": self.get_current_joints(),
                "em_movimento": self.em_movimento,
                "robot_mode": self.rtde_r.getRobotMode(),
                "safety_mode": self.rtde_r.getSafetyMode(),
                "is_program_running": self.rtde_r.isProgramRunning()
            }
            return status
        except Exception as e:
            print(f"❌ Erro ao obter status: {e}")
            return {"connected": False, "error": str(e)}

    def disconnect(self):
        """Desconecta do robô"""
        try:
            if self.rtde_c:
                self.rtde_c.stopScript()
                print("🔌 Desconectado do robô")
            self.em_movimento = False
        except Exception as e:
            print(f"❌ Erro ao desconectar: {e}")
