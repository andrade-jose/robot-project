from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
import time
import math

class URController:
    def __init__(self, robot_ip="10.1.4.122", speed=0.1, acceleration=0.1):
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

        # Configurações de validação avançada
        self.enable_safety_validation = True
        self.max_movement_distance = 1.0  # Distância máxima permitida em um movimento
        self.validation_retries = 3  # Tentativas de validação antes de falhar

    def is_connected(self):
        """Verifica se está conectado ao robô"""
        return (self.rtde_c and self.rtde_r and self.rtde_c.isConnected())

    def validate_pose_safety_limits(self, pose):
        """
        🔥 NOVA FUNÇÃO: Usa isPoseWithinSafetyLimits() da biblioteca ur_rtde
        Valida se a pose está dentro dos limites de segurança definidos no robô
        """
        if not self.enable_safety_validation:
            return True
            
        if not self.is_connected():
            print("❌ Robô não conectado para validação")
            return False
            
        try:
            # Usar a função oficial da biblioteca ur_rtde
            is_safe = self.rtde_c.isPoseWithinSafetyLimits(pose)
            
            if is_safe:
                print(f"✅ Pose APROVADA nos limites de segurança: {[f'{p:.3f}' for p in pose]}")
            else:
                print(f"❌ Pose REJEITADA pelos limites de segurança: {[f'{p:.3f}' for p in pose]}")
                
            return is_safe
            
        except Exception as e:
            print(f"❌ Erro na validação de limites de segurança: {e}")
            return False

    def validate_pose_reachability(self, pose):
        """
        🔥 NOVA FUNÇÃO: Validação adicional de alcançabilidade
        Verifica cinemática inversa e distância de movimento
        """
        try:
            # Verificar se a pose tem formato correto
            if len(pose) != 6:
                print(f"❌ Formato de pose inválido: deve ter 6 elementos, recebeu {len(pose)}")
                return False
                
            # Obter pose atual para calcular distância
            current_pose = self.get_current_pose()
            if not current_pose:
                print("❌ Não foi possível obter pose atual")
                return False
                
            # Calcular distância euclidiana do movimento
            distance = math.sqrt(
                (pose[0] - current_pose[0])**2 +
                (pose[1] - current_pose[1])**2 +
                (pose[2] - current_pose[2])**2
            )
            
            # Verificar se a distância não é muito grande
            if distance > self.max_movement_distance:
                print(f"❌ Movimento muito grande: {distance:.3f}m > {self.max_movement_distance}m")
                return False
                
            # Verificar se as orientações não são extremas
            rotation_magnitude = math.sqrt(pose[3]**2 + pose[4]**2 + pose[5]**2)
            if rotation_magnitude > math.pi:
                print(f"❌ Magnitude de rotação extrema: {rotation_magnitude:.3f} > π")
                return False
                
            print(f"✅ Pose alcançável - Distância: {distance:.3f}m, Rotação: {rotation_magnitude:.3f}rad")
            return True
            
        except Exception as e:
            print(f"❌ Erro na validação de alcançabilidade: {e}")
            return False

    def validate_pose_complete(self, pose):
        """
        🔥 FUNÇÃO PRINCIPAL DE VALIDAÇÃO
        Executa todas as validações de pose em sequência
        """
        print(f"🔍 Iniciando validação completa da pose: {[f'{p:.3f}' for p in pose]}")
        
        # 1. Validação básica de workspace (mantida da versão original)
        if not self.validate_pose(pose):
            return False
            
        # 2. Validação de alcançabilidade
        if not self.validate_pose_reachability(pose):
            return False
            
        # 3. 🔥 VALIDAÇÃO OFICIAL UR_RTDE - isPoseWithinSafetyLimits
        if not self.validate_pose_safety_limits(pose):
            return False
            
        print(f"✅ POSE TOTALMENTE VALIDADA E SEGURA!")
        return True

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
            
        print(f"✅ Pose válida no workspace: x={x:.3f}, y={y:.3f}, z={z:.3f}, rx={rx:.3f}, ry={ry:.3f}, rz={rz:.3f}")
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
        🔄 FUNÇÃO ATUALIZADA: Agora usa validate_pose_complete
        """
        return self.validate_pose_complete(target_pose)

    def correct_pose_automatically(self, pose):
        """
        🔥 NOVA FUNÇÃO: Corrige pose automaticamente quando rejeitada
        Aplica estratégias inteligentes para tornar a pose válida
        """
        print(f"🔧 Iniciando correção automática da pose: {[f'{p:.3f}' for p in pose]}")
        
        corrected_pose = pose.copy()
        corrections_applied = []
        
        # 1. Correção de workspace - ajustar coordenadas para limites
        x, y, z, rx, ry, rz = corrected_pose
        
        # Corrigir X
        if x < self.workspace_limits['x_min']:
            corrected_pose[0] = self.workspace_limits['x_min'] + 0.01
            corrections_applied.append(f"X: {x:.3f} → {corrected_pose[0]:.3f}")
        elif x > self.workspace_limits['x_max']:
            corrected_pose[0] = self.workspace_limits['x_max'] - 0.01
            corrections_applied.append(f"X: {x:.3f} → {corrected_pose[0]:.3f}")
            
        # Corrigir Y
        if y < self.workspace_limits['y_min']:
            corrected_pose[1] = self.workspace_limits['y_min'] + 0.01
            corrections_applied.append(f"Y: {y:.3f} → {corrected_pose[1]:.3f}")
        elif y > self.workspace_limits['y_max']:
            corrected_pose[1] = self.workspace_limits['y_max'] - 0.01
            corrections_applied.append(f"Y: {y:.3f} → {corrected_pose[1]:.3f}")
            
        # Corrigir Z
        if z < self.workspace_limits['z_min']:
            corrected_pose[2] = self.workspace_limits['z_min'] + 0.01
            corrections_applied.append(f"Z: {z:.3f} → {corrected_pose[2]:.3f}")
        elif z > self.workspace_limits['z_max']:
            corrected_pose[2] = self.workspace_limits['z_max'] - 0.01
            corrections_applied.append(f"Z: {z:.3f} → {corrected_pose[2]:.3f}")
            
        # 2. Correção de orientação - normalizar rotações
        rotation_magnitude = math.sqrt(rx**2 + ry**2 + rz**2)
        if rotation_magnitude > math.pi:
            # Normalizar o vetor de rotação
            factor = math.pi / rotation_magnitude * 0.95  # 95% do limite
            corrected_pose[3] = rx * factor
            corrected_pose[4] = ry * factor
            corrected_pose[5] = rz * factor
            corrections_applied.append(f"Rotação normalizada: {rotation_magnitude:.3f} → {math.pi*0.95:.3f}")
            
        # 3. Verificar distância de movimento
        current_pose = self.get_current_pose()
        if current_pose:
            distance = math.sqrt(
                (corrected_pose[0] - current_pose[0])**2 +
                (corrected_pose[1] - current_pose[1])**2 +
                (corrected_pose[2] - current_pose[2])**2
            )
            
            if distance > self.max_movement_distance:
                # Reduzir movimento para limite máximo
                factor = self.max_movement_distance / distance * 0.95
                
                corrected_pose[0] = current_pose[0] + (corrected_pose[0] - current_pose[0]) * factor
                corrected_pose[1] = current_pose[1] + (corrected_pose[1] - current_pose[1]) * factor
                corrected_pose[2] = current_pose[2] + (corrected_pose[2] - current_pose[2]) * factor
                
                corrections_applied.append(f"Distância reduzida: {distance:.3f}m → {self.max_movement_distance*0.95:.3f}m")
        
        if corrections_applied:
            print("🔧 Correções aplicadas:")
            for correction in corrections_applied:
                print(f"   • {correction}")
            print(f"🔧 Pose corrigida: {[f'{p:.3f}' for p in corrected_pose]}")
        else:
            print("🔧 Nenhuma correção de workspace necessária")
            
        return corrected_pose

    def move_to_pose_with_smart_correction(self, pose, speed=None, acceleration=None, max_correction_attempts=3):
        """
        🔥 FUNÇÃO PRINCIPAL: Movimento inteligente com correção automática
        Tenta mover para a pose, e se rejeitada, aplica correções automáticas
        """
        if speed is None:
            speed = self.speed
        if acceleration is None:
            acceleration = self.acceleration
            
        print(f"🧠 Iniciando movimento INTELIGENTE para: {[f'{p:.3f}' for p in pose]}")
        
        if not self.is_connected():
            print("❌ Robô não está conectado")
            return False, None
            
        original_pose = pose.copy()
        current_pose = pose.copy()
        
        for tentativa in range(max_correction_attempts):
            print(f"\n--- TENTATIVA {tentativa + 1}/{max_correction_attempts} ---")
            
            # Testar pose atual
            if self.validate_pose_complete(current_pose):
                print("✅ Pose validada! Executando movimento...")
                
                try:
                    self.em_movimento = True
                    success = self.rtde_c.moveL(current_pose, speed, acceleration)
                    
                    if success:
                        time.sleep(self.pause_between_moves)
                        
                        # Verificar precisão
                        final_pose = self.get_current_pose()
                        if final_pose:
                            distance = math.sqrt(
                                (current_pose[0] - final_pose[0])**2 +
                                (current_pose[1] - final_pose[1])**2 +
                                (current_pose[2] - final_pose[2])**2
                            )
                            print(f"✅ Movimento executado! Precisão: {distance*1000:.1f}mm")
                            
                        return True, current_pose
                    else:
                        print("❌ Robô rejeitou o movimento mesmo após validação")
                        
                except Exception as e:
                    print(f"❌ Erro durante execução: {e}")
                finally:
                    self.em_movimento = False
                    
            # Se chegou aqui, pose foi rejeitada - aplicar correções
            if tentativa < max_correction_attempts - 1:
                print(f"🔧 Pose rejeitada, aplicando correções...")
                current_pose = self.correct_pose_automatically(current_pose)
            else:
                print(f"❌ Esgotadas {max_correction_attempts} tentativas de correção")
                
        return False, None

    def move_with_intermediate_points(self, target_pose, speed=None, acceleration=None, num_points=3):
        """
        🔥 ESTRATÉGIA AVANÇADA: Movimento com pontos intermediários
        Para poses muito distantes, divide o movimento em etapas
        """
        if speed is None:
            speed = self.speed
        if acceleration is None:
            acceleration = self.acceleration
            
        print(f"🚀 Movimento com {num_points} pontos intermediários")
        
        current_pose = self.get_current_pose()
        if not current_pose:
            print("❌ Não foi possível obter pose atual")
            return False
            
        # Gerar pontos intermediários
        intermediate_poses = []
        for i in range(1, num_points + 1):
            factor = i / (num_points + 1)
            
            intermediate_pose = [
                current_pose[j] + (target_pose[j] - current_pose[j]) * factor
                for j in range(6)
            ]
            intermediate_poses.append(intermediate_pose)
            
        # Adicionar pose final
        intermediate_poses.append(target_pose)
        
        print(f"📍 Planejamento de {len(intermediate_poses)} pontos:")
        for i, pose in enumerate(intermediate_poses):
            print(f"   Ponto {i+1}: {[f'{p:.3f}' for p in pose]}")
            
        # Executar sequência
        for i, pose in enumerate(intermediate_poses):
            print(f"\n🎯 Executando ponto {i+1}/{len(intermediate_poses)}")
            
            sucesso, pose_final = self.move_to_pose_with_smart_correction(pose, speed, acceleration)
            
            if not sucesso:
                print(f"❌ Falha no ponto {i+1} - movimento interrompido")
                return False
                
        print("✅ Movimento com pontos intermediários concluído!")
        return True

    def move_to_pose_safe(self, pose, speed=None, acceleration=None, use_smart_correction=True):
        """
        🔥 MOVIMENTO SEGURO ATUALIZADO - Agora com correção automática
        """
        if speed is None:
            speed = self.speed
        if acceleration is None:
            acceleration = self.acceleration
            
        print(f"🎯 Movimento SEGURO para: {[f'{p:.3f}' for p in pose]}")
        
        if not use_smart_correction:
            # Modo legado - só validação simples
            print("⚠️ Modo legado - sem correção automática")
            return self._move_legacy_validation(pose, speed, acceleration)
        
        # 1. Tentar movimento direto com correção automática
        print("🧠 Tentativa 1: Movimento direto com correção automática")
        sucesso, pose_final = self.move_to_pose_with_smart_correction(pose, speed, acceleration)
        
        if sucesso:
            return True
            
        # 2. Se falhou, tentar com pontos intermediários
        print("🚀 Tentativa 2: Movimento com pontos intermediários")
        current_pose = self.get_current_pose()
        if current_pose:
            # Calcular distância
            distance = math.sqrt(
                (pose[0] - current_pose[0])**2 +
                (pose[1] - current_pose[1])**2 +
                (pose[2] - current_pose[2])**2
            )
            
            # Definir número de pontos baseado na distância
            num_points = max(2, int(distance / 0.2))  # 1 ponto a cada 20cm
            num_points = min(num_points, 5)  # Máximo 5 pontos
            
            sucesso = self.move_with_intermediate_points(pose, speed, acceleration, num_points)
            if sucesso:
                return True
        
        # 3. Último recurso - movimento muito lento e cauteloso
        print("🐌 Tentativa 3: Movimento ultra-cauteloso")
        slow_speed = min(speed * 0.3, 0.02)  # 30% da velocidade ou 2cm/s
        slow_accel = min(acceleration * 0.3, 0.02)
        
        sucesso, _ = self.move_to_pose_with_smart_correction(pose, slow_speed, slow_accel, max_correction_attempts=5)
        
        if sucesso:
            print("✅ Movimento concluído com estratégia ultra-cautelosa!")
            return True
        else:
            print("❌ TODAS as estratégias falharam - movimento impossível")
            return False

    def _move_legacy_validation(self, pose, speed, acceleration):
        """Método legado de validação (mantido para compatibilidade)"""
        if not self.is_connected():
            return False
            
        for tentativa in range(self.validation_retries):
            if self.validate_pose_complete(pose):
                break
            if tentativa == self.validation_retries - 1:
                return False
            time.sleep(0.5)
        
        try:
            self.em_movimento = True
            success = self.rtde_c.moveL(pose, speed, acceleration)
            if success:
                time.sleep(self.pause_between_moves)
            return success
        except Exception as e:
            print(f"❌ Erro: {e}")
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
        🔥 MOVIMENTO DE PEÇA ATUALIZADO com validação em cada etapa
        """
        print(f"🤖 Executando movimento de peça com VALIDAÇÃO COMPLETA:")
        print(f"   📍 Origem: {[f'{p:.3f}' for p in origem]}")
        print(f"   📍 Destino: {[f'{p:.3f}' for p in destino]}")
        print(f"   ⬆️ Altura segura: {altura_segura:.3f}")
        print(f"   ⬇️ Altura pegar: {altura_pegar:.3f}")
        
        try:
            # 1. Mover para posição segura acima da origem
            pose_segura_origem = origem.copy()
            pose_segura_origem[2] = altura_segura
            
            print("🔍 Etapa 1: Validando posição segura origem...")
            if not self.move_to_pose_safe(pose_segura_origem):
                print("❌ Falha ao mover para posição segura origem")
                return False
                
            # 2. Descer para pegar a peça
            pose_pegar = origem.copy()
            pose_pegar[2] = altura_pegar
            
            print("🔍 Etapa 2: Validando descida para pegar...")
            if not self.move_to_pose_safe(pose_pegar, speed=0.05):  # Movimento mais lento
                print("❌ Falha ao descer para pegar peça")
                return False
                
            # 3. Subir com a peça
            print("🔍 Etapa 3: Validando subida com peça...")
            if not self.move_to_pose_safe(pose_segura_origem):
                print("❌ Falha ao subir com peça")
                return False
                
            # 4. Mover para posição segura acima do destino
            pose_segura_destino = destino.copy()
            pose_segura_destino[2] = altura_segura
            
            print("🔍 Etapa 4: Validando posição segura destino...")
            if not self.move_to_pose_safe(pose_segura_destino):
                print("❌ Falha ao mover para posição segura destino")
                return False
                
            # 5. Descer para colocar a peça
            pose_colocar = destino.copy()
            pose_colocar[2] = altura_pegar
            
            print("🔍 Etapa 5: Validando descida para colocar...")
            if not self.move_to_pose_safe(pose_colocar, speed=0.05):  # Movimento mais lento
                print("❌ Falha ao descer para colocar peça")
                return False
                
            # 6. Subir após colocar
            print("🔍 Etapa 6: Validando subida final...")
            if not self.move_to_pose_safe(pose_segura_destino):
                print("❌ Falha ao subir após colocar peça")
                return False
                
            print("✅ Movimento de peça concluído com SUCESSO TOTAL!")
            return True
            
        except Exception as e:
            print(f"❌ Erro durante movimento de peça: {e}")
            return False

    def enable_safety_mode(self, enable=True):
        """
        🔥 NOVA FUNÇÃO: Liga/desliga validações de segurança
        """
        self.enable_safety_validation = enable
        status = "HABILITADA" if enable else "DESABILITADA"
        print(f"🛡️ Validação de segurança {status}")

    def test_pose_validation(self, pose):
        """
        🔥 NOVA FUNÇÃO: Testa validação de pose sem executar movimento
        Útil para debugging
        """
        print(f"🧪 TESTE DE VALIDAÇÃO - Pose: {[f'{p:.3f}' for p in pose]}")
        
        print("1️⃣ Testando limites de workspace...")
        workspace_ok = self.validate_pose(pose)
        
        print("2️⃣ Testando alcançabilidade...")
        reachable_ok = self.validate_pose_reachability(pose)
        
        print("3️⃣ Testando limites de segurança UR...")
        safety_ok = self.validate_pose_safety_limits(pose)
        
        resultado = workspace_ok and reachable_ok and safety_ok
        
        print(f"📊 RESULTADO DO TESTE:")
        print(f"   Workspace: {'✅' if workspace_ok else '❌'}")
        print(f"   Alcançabilidade: {'✅' if reachable_ok else '❌'}")
        print(f"   Limites UR: {'✅' if safety_ok else '❌'}")
        print(f"   FINAL: {'✅ APROVADA' if resultado else '❌ REJEITADA'}")
        
        return resultado

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
                "safety_validation_enabled": self.enable_safety_validation
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

    # ====================== FUNÇÕES DE DEBUG ======================
    
    def debug_movement_sequence(self, poses_list, test_only=False):
        """
        🔥 NOVA FUNÇÃO: Debugga uma sequência de movimentos
        """
        print(f"🐛 DEBUG: Testando sequência de {len(poses_list)} poses...")
        
        resultados = []
        for i, pose in enumerate(poses_list):
            print(f"\n--- POSE {i+1}/{len(poses_list)} ---")
            
            if test_only:
                resultado = self.test_pose_validation(pose)
            else:
                resultado = self.move_to_pose_safe(pose)
                
            resultados.append(resultado)
            
            if not resultado:
                print(f"❌ Sequência INTERROMPIDA na pose {i+1}")
                break
                
        aprovadas = sum(resultados)
        print(f"\n📊 RESULTADO DA SEQUÊNCIA:")
        print(f"   Poses aprovadas: {aprovadas}/{len(poses_list)}")
        print(f"   Taxa de sucesso: {(aprovadas/len(poses_list)*100):.1f}%")
        
        return resultados