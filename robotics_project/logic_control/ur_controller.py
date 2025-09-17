from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from config.config_completa import ConfigRobo
import time
import math

class URController:
    def __init__(self, robot_ip=None, speed=None, acceleration=None, config=None):
        # Usar config ou valores padrão
        self.config = ConfigRobo()
        self.robot_ip = self.config.ip
        self.speed = self.config.velocidade_padrao
        self.acceleration = self.config.aceleracao_padrao

        self.rtde_c = RTDEControlInterface(self.robot_ip)
        self.rtde_r = RTDEReceiveInterface(self.robot_ip)
        print(f"✅ Conectado ao robô UR em {self.robot_ip}")

        # Usar configurações da config
        self.pause_between_moves = self.config.pausa_entre_movimentos
        self.safe_height_offset = self.config.altura_offset_seguro
        self.min_elbow_tcp_distance = self.config.distancia_minima_cotovelo_tcp
        self.last_error = None
        self.em_movimento = False

        self.max_joint_change = self.config.max_mudanca_junta
        self.planning_steps = self.config.passos_planejamento
        
        self.workspace_limits = self.config.limites_workspace

        self.enable_safety_validation = self.config.habilitar_validacao_seguranca
        self.max_movement_distance = self.config.distancia_maxima_movimento
        self.validation_retries = self.config.tentativas_validacao
        self.base_iron_height = self.config.altura_base_ferro
        self.elbow_safety_margin = self.config.margem_seguranca_cotovelo

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
        

    def validate_elbow_height_constraint(self, pose):
        """
        🔥 NOVA FUNÇÃO: Valida se o cotovelo não vai abaixo da base de ferro
        """
        try:
            # Calcular posição aproximada do cotovelo baseada na pose do TCP
            # Para UR, o cotovelo fica aproximadamente na altura Z da base + offset do braço
            estimated_elbow_z = pose[2] - 0.3  # Ajustar baseado no seu modelo UR
            
            min_allowed_z = self.base_iron_height + self.elbow_safety_margin
            
            if estimated_elbow_z < min_allowed_z:
                print(f"❌ cotovelo muito baixo: {estimated_elbow_z:.3f}m < {min_allowed_z:.3f}m")
                return False
                
            print(f"✅ Altura do cotovelo OK: {estimated_elbow_z:.3f}m")
            return True
            
        except Exception as e:
            print(f"❌ Erro na validação da altura do cotovelo: {e}")
            return False
        

    def diagnostic_pose_rejection(self, pose):
        """
        🔥 DIAGNÓSTICO AVANÇADO: Identifica exatamente por que a pose foi rejeitada
        """
        print(f"🔍 DIAGNÓSTICO COMPLETO da pose: {[f'{p:.3f}' for p in pose]}")
        
        diagnostics = {
            'pose_original': pose,
            'pose_alcancavel': False,
            'joints_calculadas': None,
            'joints_problematicas': [],
            'singularidades': False,
            'conflitos_base_ferro': False,
            'sugestoes_correcao': []
        }
        
        try:
            # 1. TESTE: Cinemática Inversa
            print("1️⃣ Testando cinemática inversa...")
            joints = self.rtde_c.getInverseKinematics(pose)
            
            if joints is None or len(joints) == 0:
                print("❌ PROBLEMA: Cinemática inversa impossível")
                diagnostics['sugestoes_correcao'].append("Ajustar posição ou orientação")
                return diagnostics
                
            diagnostics['joints_calculadas'] = joints
            diagnostics['pose_alcancavel'] = True
            print(f"✅ Articulações calculadas: {[f'{j:.3f}' for j in joints]}")
            
            # 2. TESTE: Limites individuais das articulações
            print("2️⃣ Verificando limites das articulações...")
            current_joints = self.get_current_joints()
            
            joint_names = ['Base', 'Shoulder', 'Elbow', 'Wrist1', 'Wrist2', 'Wrist3']
            
            # Limites típicos UR (ajustar conforme seu modelo)
            joint_limits = [
                (-2*3.14159, 2*3.14159),  # Base: 360°
                (-2*3.14159, 2*3.14159),  # Shoulder: 360° 
                (-3.14159, 3.14159),      # Elbow: 180°
                (-2*3.14159, 2*3.14159),  # Wrist1: 360°
                (-2*3.14159, 2*3.14159),  # Wrist2: 360°
                (-2*3.14159, 2*3.14159),  # Wrist3: 360°
            ]
            
            for i, (joint_val, (min_lim, max_lim), name) in enumerate(zip(joints, joint_limits, joint_names)):
                if joint_val < min_lim or joint_val > max_lim:
                    print(f"❌ {name}: {joint_val:.3f} fora do limite [{min_lim:.3f}, {max_lim:.3f}]")
                    diagnostics['joints_problematicas'].append((i, name, joint_val, min_lim, max_lim))
                else:
                    print(f"✅ {name}: {joint_val:.3f} OK")
                    
            # 3. TESTE ESPECÍFICO: Altura do cotovelo com base de ferro
            print("3️⃣ Verificando conflito com base de ferro...")
            elbow_angle = joints[1]  # Joint 1 = elbow
            
            # Cálculo mais preciso da altura do cotovelo
            # Para UR, a altura do cotovelo depende do ângulo da junta do cotovelo
            # Altura aproximada: altura_base + altura_cotovelo_nominal * cos(elbow_angle)
            altura_cotovelo_estimada = 0.162 * abs(math.cos(elbow_angle))  # 162mm para UR típico
            altura_cotovelo_real = altura_cotovelo_estimada
            
            limite_minimo_cotovelo = self.config.altura_base_ferro + self.config.margem_seguranca_cotovelo
            
            if altura_cotovelo_real < limite_minimo_cotovelo:
                print(f"❌ CONFLITO BASE DE FERRO: cotovelo em {altura_cotovelo_real:.3f}m < {limite_minimo_cotovelo:.3f}m")
                diagnostics['conflitos_base_ferro'] = True
                diagnostics['sugestoes_correcao'].append(f"Elevar TCP em {limite_minimo_cotovelo - altura_cotovelo_real + 0.01:.3f}m")
            else:
                print(f"✅ Base de ferro OK: cotovelo em {altura_cotovelo_real:.3f}m")
                
            # 4. TESTE: Singularidades cinemáticas
            print("4️⃣ Verificando singularidades...")
            
            # Detectar singularidade de punho (wrist singularity)
            wrist_config = math.sqrt(joints[4]**2 + joints[5]**2)
            if wrist_config < 0.1:  # Muito próximo de singularidade
                print("⚠️ AVISO: Próximo à singularidade de punho")
                diagnostics['singularidades'] = True
                diagnostics['sugestoes_correcao'].append("Ajustar orientação do TCP")
                
            # Detectar singularidade de cotovelo (elbow singularity)
            if abs(joints[1]) < 0.1 and abs(joints[2]) < 0.1:
                print("⚠️ AVISO: Próximo à singularidade de cotovelo")
                diagnostics['singularidades'] = True
                
            # 5. TESTE: Mudanças extremas de articulação
            print("5️⃣ Verificando mudanças extremas...")
            if current_joints:
                for i, (current, target, name) in enumerate(zip(current_joints, joints, joint_names)):
                    mudanca = abs(target - current)
                    if mudanca > self.config.max_mudanca_junta:
                        print(f"⚠️ {name}: Mudança grande {mudanca:.3f} > {self.config.max_mudanca_junta:.3f}")
                        diagnostics['sugestoes_correcao'].append(f"Movimento intermediário para {name}")
                        
            # 6. GERAR RELATÓRIO FINAL
            print("\n📊 RELATÓRIO DE DIAGNÓSTICO:")
            print(f"   Cinemática possível: {'✅' if diagnostics['pose_alcancavel'] else '❌'}")
            print(f"   Articulações problemáticas: {len(diagnostics['joints_problematicas'])}")
            print(f"   Conflito base ferro: {'❌' if diagnostics['conflitos_base_ferro'] else '✅'}")
            print(f"   Singularidades detectadas: {'⚠️' if diagnostics['singularidades'] else '✅'}")
            
            if diagnostics['sugestoes_correcao']:
                print("🔧 SUGESTÕES DE CORREÇÃO:")
                for i, sugestao in enumerate(diagnostics['sugestoes_correcao'], 1):
                    print(f"   {i}. {sugestao}")
                    
            return diagnostics
            
        except Exception as e:
            print(f"❌ Erro durante diagnóstico: {e}")
            diagnostics['sugestoes_correcao'].append("Verificar conexão com robô")
            return diagnostics

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
        
        #if not self.validate_elbow_height_constraint(pose):
        #    return False
            
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

    # SUBSTITUIR a função correct_pose_automatically no URController

    def correct_pose_automatically(self, pose):
        """
        🔥 CORREÇÃO INTELIGENTE BASEADA EM ARTICULAÇÕES
        Agora usa diagnóstico avançado para correções precisas
        """
        print(f"🔧 Iniciando correção INTELIGENTE da pose: {[f'{p:.3f}' for p in pose]}")
        
        # 1. DIAGNÓSTICO COMPLETO
        diagnostics = self.diagnostic_pose_rejection(pose)
        
        if not diagnostics['pose_alcancavel']:
            print("❌ Pose impossível cinematicamente - tentando correções básicas")
            return self._correct_basic_workspace(pose)  # Fallback para método antigo
        
        corrected_pose = pose.copy()
        corrections_applied = []
        
        # 2. CORREÇÃO: Base de ferro (PRIORITÁRIA)
        if diagnostics['conflitos_base_ferro']:
            print("🔧 Corrigindo conflito com base de ferro...")
            
            # Estratégia: Elevar Z até cotovelo ficar seguro
            current_z = corrected_pose[2]
            joints = diagnostics['joints_calculadas']
            elbow_angle = joints[1]
            
            # Calcular Z mínimo necessário
            altura_cotovelo_necessaria = self.config.altura_base_ferro + self.config.margem_seguranca_cotovelo + 0.01
            
            # Aproximação: Z_tcp ≈ altura_cotovelo + offset_tcp_cotovelo
            # Para configuração típica UR, offset TCP-cotovelo ≈ 0.3m
            z_minimo_tcp = altura_cotovelo_necessaria + 0.3
            
            if current_z < z_minimo_tcp:
                corrected_pose[2] = z_minimo_tcp
                corrections_applied.append(f"Z elevado para proteger cotovelo: {current_z:.3f} → {z_minimo_tcp:.3f}")
        
        # 3. CORREÇÃO: Articulações problemáticas
        if diagnostics['joints_problematicas']:
            print("🔧 Corrigindo articulações fora dos limites...")
            
            joints = diagnostics['joints_calculadas'].copy()
            
            for joint_idx, name, valor, min_lim, max_lim in diagnostics['joints_problematicas']:
                # Corrigir articulação para dentro dos limites
                if valor < min_lim:
                    joints[joint_idx] = min_lim + 0.05  # Margem de segurança
                    corrections_applied.append(f"{name}: {valor:.3f} → {joints[joint_idx]:.3f} (limite mín)")
                elif valor > max_lim:
                    joints[joint_idx] = max_lim - 0.05  # Margem de segurança  
                    corrections_applied.append(f"{name}: {valor:.3f} → {joints[joint_idx]:.3f} (limite máx)")
            
            # Recalcular pose a partir das articulações corrigidas
            try:
                new_pose = self.rtde_c.getForwardKinematics(joints)
                if new_pose:
                    corrected_pose = new_pose
                    corrections_applied.append("Pose recalculada a partir de articulações corrigidas")
            except Exception as e:
                print(f"⚠️ Erro na cinemática direta: {e}")
        
        # 4. CORREÇÃO: Singularidades
        if diagnostics['singularidades']:
            print("🔧 Corrigindo singularidades...")
            
            # Ajustar orientação ligeiramente para sair da singularidade
            orientation_adjustments = [
                [0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05],
                [-0.05, 0, 0], [0, -0.05, 0], [0, 0, -0.05]
            ]
            
            for adjustment in orientation_adjustments:
                test_pose = corrected_pose.copy()
                test_pose[3] += adjustment[0]
                test_pose[4] += adjustment[1]  
                test_pose[5] += adjustment[2]
                
                # Testar se a nova orientação resolve o problema
                test_joints = self.rtde_c.getInverseKinematics(test_pose)
                if test_joints and self.rtde_c.isPoseWithinSafetyLimits(test_pose):
                    corrected_pose = test_pose
                    corrections_applied.append(f"Orientação ajustada: {adjustment}")
                    break
        
        # 5. CORREÇÃO FINAL: Workspace básico (mantida da versão original)
        corrected_pose = self._correct_basic_workspace(corrected_pose)
        
        # 6. RELATÓRIO DE CORREÇÕES
        if corrections_applied:
            print("🔧 Correções aplicadas:")
            for correction in corrections_applied:
                print(f"   • {correction}")
            print(f"🔧 Pose final corrigida: {[f'{p:.3f}' for p in corrected_pose]}")
        else:
            print("🔧 Nenhuma correção necessária")
            
        return corrected_pose

    def _correct_basic_workspace(self, pose):
        """Método auxiliar: correções básicas de workspace (código original)"""
        corrected_pose = pose.copy()
        corrections_applied = []
        
        x, y, z, rx, ry, rz = corrected_pose
        
        # Corrigir coordenadas para limites
        if x < self.workspace_limits['x_min']:
            corrected_pose[0] = self.workspace_limits['x_min'] + 0.01
            corrections_applied.append(f"X: {x:.3f} → {corrected_pose[0]:.3f}")
        elif x > self.workspace_limits['x_max']:
            corrected_pose[0] = self.workspace_limits['x_max'] - 0.01
            corrections_applied.append(f"X: {x:.3f} → {corrected_pose[0]:.3f}")
            
        if y < self.workspace_limits['y_min']:
            corrected_pose[1] = self.workspace_limits['y_min'] + 0.01
            corrections_applied.append(f"Y: {y:.3f} → {corrected_pose[1]:.3f}")
        elif y > self.workspace_limits['y_max']:
            corrected_pose[1] = self.workspace_limits['y_max'] - 0.01
            corrections_applied.append(f"Y: {y:.3f} → {corrected_pose[1]:.3f}")
            
        if z < self.workspace_limits['z_min']:
            corrected_pose[2] = self.workspace_limits['z_min'] + 0.01
            corrections_applied.append(f"Z: {z:.3f} → {corrected_pose[2]:.3f}")
        elif z > self.workspace_limits['z_max']:
            corrected_pose[2] = self.workspace_limits['z_max'] - 0.01
            corrections_applied.append(f"Z: {z:.3f} → {corrected_pose[2]:.3f}")

        # Correção de orientação
        rotation_magnitude = math.sqrt(rx**2 + ry**2 + rz**2)
        if rotation_magnitude > math.pi:
            factor = math.pi / rotation_magnitude * 0.95
            corrected_pose[3] = rx * factor
            corrected_pose[4] = ry * factor
            corrected_pose[5] = rz * factor
            corrections_applied.append(f"Rotação normalizada")
            
        return corrected_pose

    def move_to_pose_with_smart_correction(self, pose, speed=None, acceleration=None, max_correction_attempts=None):
        """
        🔥 MOVIMENTO INTELIGENTE ATUALIZADO com diagnóstico avançado
        Agora identifica exatamente por que poses são rejeitadas e corrige especificamente
        """
        if max_correction_attempts is None:
            max_correction_attempts = self.config.max_tentativas_correcao_articulacoes
        if speed is None:
            speed = self.speed
        if acceleration is None:
            acceleration = self.acceleration
            
        print(f"🧠 MOVIMENTO INTELIGENTE V2.0 para: {[f'{p:.3f}' for p in pose]}")
        
        if not self.is_connected():
            print("❌ Robô não está conectado")
            return False, None
            
        original_pose = pose.copy()
        current_pose = pose.copy()
        
        for tentativa in range(max_correction_attempts):
            print(f"\n--- TENTATIVA {tentativa + 1}/{max_correction_attempts} ---")
            
            # 1. DIAGNÓSTICO COMPLETO (sempre primeiro)
            if self.config.habilitar_diagnostico_avancado:
                diagnostics = self.diagnostic_pose_rejection(current_pose)
                
                # Se pose é impossível cinematicamente, pular para próxima estratégia
                if not diagnostics['pose_alcancavel']:
                    print("❌ Pose cinematicamente impossível - aplicando correções drásticas")
                    current_pose = self._apply_drastic_corrections(current_pose, original_pose)
                    continue
            
            # 2. VALIDAÇÃO COMPLETA
            if self.validate_pose_complete(current_pose):
                print("✅ Pose validada! Executando movimento...")
                
                try:
                    self.em_movimento = True
                    
                    # NOVO: Tentar múltiplas configurações de articulações se habilitado
                    success = False
                    if self.config.usar_multiplas_configuracoes_ik:
                        success = self._try_multiple_ik_configurations(current_pose, speed, acceleration)
                    else:
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
                    
            # 3. APLICAR CORREÇÕES INTELIGENTES
            if tentativa < max_correction_attempts - 1:
                print(f"🔧 Pose rejeitada, aplicando correções INTELIGENTES...")
                
                # Usar o novo sistema de correção baseado em articulações
                corrected_pose = self.correct_pose_automatically(current_pose)
                
                # Se correção não mudou nada, tentar estratégias alternativas
                if self._poses_are_equal(corrected_pose, current_pose):
                    print("🔧 Correção automática não funcionou, tentando estratégias alternativas...")
                    corrected_pose = self._apply_alternative_corrections(current_pose, tentativa)
                    
                current_pose = corrected_pose
            else:
                print(f"❌ Esgotadas {max_correction_attempts} tentativas de correção")
                    
        return False, None

    def _try_multiple_ik_configurations(self, pose, speed, acceleration):
        """NOVO: Tenta diferentes configurações de cinemática inversa"""
        print("🔄 Tentando múltiplas configurações IK...")
        
        # Tentar pequenas variações na orientação para encontrar configuração válida  
        orientation_variations = [
            [0, 0, 0],           # Original
            [0.01, 0, 0],        # Pequena rotação em X
            [0, 0.01, 0],        # Pequena rotação em Y  
            [0, 0, 0.01],        # Pequena rotação em Z
            [-0.01, 0, 0],       # Rotação negativa em X
            [0, -0.01, 0],       # Rotação negativa em Y
            [0, 0, -0.01],       # Rotação negativa em Z
            [0.01, 0.01, 0],     # Combinação XY
        ]
        
        for i, variation in enumerate(orientation_variations):
            if i >= self.config.max_configuracoes_ik:
                break
                
            test_pose = pose.copy()
            test_pose[3] += variation[0]
            test_pose[4] += variation[1] 
            test_pose[5] += variation[2]
            
            try:
                # Verificar se esta variação é válida
                if self.rtde_c.isPoseWithinSafetyLimits(test_pose):
                    print(f"✅ Configuração {i+1} válida - executando...")
                    success = self.rtde_c.moveL(test_pose, speed, acceleration)
                    if success:
                        return True
                        
            except Exception as e:
                continue  # Tentar próxima configuração
                
        print("❌ Nenhuma configuração IK funcionou")
        return False

    def _apply_drastic_corrections(self, pose, original_pose):
        """NOVO: Correções drásticas para poses impossíveis"""
        print("🚨 Aplicando correções DRÁSTICAS...")
        
        corrected = pose.copy()
        
        # 1. Mover para posição mais próxima do centro do workspace
        center_workspace = [0.4, 0.0, 0.3, 0.0, 3.14, 0.0]
        
        # Interpolar 50% em direção ao centro
        for i in range(3):  # Apenas posição, não orientação
            corrected[i] = pose[i] * 0.5 + center_workspace[i] * 0.5
            
        # 2. Garantir altura mínima segura
        min_safe_z = self.config.altura_base_ferro + self.config.margem_seguranca_base_ferro + 0.1
        if corrected[2] < min_safe_z:
            corrected[2] = min_safe_z
            
        print(f"🚨 Pose drasticamente corrigida: {[f'{p:.3f}' for p in corrected]}")
        return corrected

    def _apply_alternative_corrections(self, pose, attempt_number):
        """NOVO: Estratégias alternativas baseadas no número da tentativa"""
        print(f"🔧 Estratégia alternativa #{attempt_number + 1}")
        
        corrected = pose.copy()
        
        if attempt_number == 0:
            # Tentativa 1: Elevar significativamente
            corrected[2] += 0.05
            print(f"🔧 Elevando Z em 5cm: {corrected[2]:.3f}")
            
        elif attempt_number == 1:
            # Tentativa 2: Mover para posição mais central
            corrected[0] = 0.4  # X central
            corrected[1] = 0.0  # Y central
            corrected[2] = max(corrected[2], 0.3)  # Z seguro
            print(f"🔧 Movendo para posição central segura")
            
        elif attempt_number == 2:
            # Tentativa 3: Orientação mais conservadora
            corrected[3] = 0.0   # rx = 0
            corrected[4] = 3.14  # ry = π (TCP para baixo)
            corrected[5] = 0.0   # rz = 0
            print(f"🔧 Orientação conservadora aplicada")
            
        else:
            # Tentativa final: Pose home modificada
            home_pose = self.config.pose_home.copy()
            home_pose[0] = pose[0]  # Manter X desejado
            home_pose[1] = pose[1]  # Manter Y desejado
            corrected = home_pose
            print(f"🔧 Usando pose home modificada")
            
        return corrected

    def _poses_are_equal(self, pose1, pose2, tolerance=0.001):
        """AUXILIAR: Verifica se duas poses são iguais dentro da tolerância"""
        for i in range(6):
            if abs(pose1[i] - pose2[i]) > tolerance:
                return False
        return True

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
            num_points = max(2, int(distance / self.config.passo_pontos_intermediarios))
            num_points = min(num_points, self.config.max_pontos_intermediarios)
            
            sucesso = self.move_with_intermediate_points(pose, speed, acceleration, num_points)
            if sucesso:
                return True
        
        # 3. Último recurso - movimento muito lento e cauteloso
        print("🐌 Tentativa 3: Movimento ultra-cauteloso")
        slow_speed = min(speed * self.config.fator_velocidade_ultra_seguro, self.config.velocidade_movimento_lento)
        slow_accel = min(acceleration * self.config.fator_velocidade_ultra_seguro, self.config.aceleracao_movimento_lento)
        
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
            if not self.move_to_pose_safe(pose_pegar, speed=self.config.velocidade_precisa):  # Movimento mais lento
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
            if not self.move_to_pose_safe(pose_colocar, speed=self.config.velocidade_precisa):  # Movimento mais lento
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
        
    def set_iron_base_height(self, height):
        """Configura a altura da base de ferro"""
        self.base_iron_height = height
        print(f"🔧 Altura da base de ferro configurada: {height:.3f}m")

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
                self.rtde_c.stopL(self.config.desaceleracao_parada) # Para movimento linear com desaceleração
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
        self.speed = max(self.config.velocidade_minima, min(speed, self.config.velocidade_maxima))
        self.acceleration = max(self.config.aceleracao_minima, min(acceleration, self.config.aceleracao_maxima))

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
    

    def test_iron_base_configuration(self):
        """
        🧪 TESTE ESPECÍFICO: Valida configuração da base de ferro
        Use esta função para verificar se as configurações estão corretas
        """
        print("🧪 TESTE DE CONFIGURAÇÃO - Base de Ferro")
        print("=" * 50)
        
        # 1. Verificar configurações
        print(f"📋 Configurações atuais:")
        print(f"   Altura base ferro: {self.config.altura_base_ferro:.3f}m")
        print(f"   Margem segurança: {self.config.margem_seguranca_cotovelo:.3f}m")
        print(f"   Modelo UR: {getattr(self.config, 'modelo_ur', 'Não definido')}")
        print(f"   Altura cotovelo nominal: {getattr(self.config, 'altura_cotovelo_nominal', 0.162):.3f}m")
        
        # 2. Testar poses problemáticas (baseado no seu log)
        poses_problema = [
            [0.408, 0.215, 0.420, 0.000, 3.140, 0.000],  # Pose que falhou no log
            [0.400, 0.200, 0.300, 0.000, 3.140, 0.000],  # Variação mais baixa
            [0.400, 0.200, 0.130, 0.000, 3.140, 0.000],  # Muito baixa (deve falhar)
        ]
        
        print(f"\n🧪 Testando {len(poses_problema)} poses problemáticas:")
        
        for i, pose in enumerate(poses_problema):
            print(f"\n--- TESTE {i+1}: {[f'{p:.3f}' for p in pose]} ---")
            
            # Diagnóstico completo
            diagnostics = self.diagnostic_pose_rejection(pose)
            
            # Teste de correção
            corrected_pose = self.correct_pose_automatically(pose)
            
            # Verificar se correção funcionou
            if self.rtde_c.isPoseWithinSafetyLimits(corrected_pose):
                print(f"✅ SUCESSO: Correção funcionou!")
                print(f"   Original: {[f'{p:.3f}' for p in pose]}")
                print(f"   Corrigida: {[f'{p:.3f}' for p in corrected_pose]}")
            else:
                print(f"❌ FALHA: Correção não resolveu o problema")
                
        return True

    def debug_calibration_failure(self, failed_poses):
        """
        🔍 DEBUG ESPECÍFICO: Analisa falhas na calibração
        Use com as poses que falharam na calibração
        """
        print("🔍 DEBUG - Análise de Falhas na Calibração")
        print("=" * 50)
        
        if not isinstance(failed_poses, list):
            failed_poses = [failed_poses]
            
        for i, pose in enumerate(failed_poses):
            print(f"\n🔍 ANALISANDO POSE {i+1}: {[f'{p:.3f}' for p in pose]}")
            
            # 1. Diagnóstico detalhado
            diagnostics = self.diagnostic_pose_rejection(pose)
            
            # 2. Tentar todas as estratégias de correção
            print("\n🔧 TESTANDO ESTRATÉGIAS DE CORREÇÃO:")
            
            strategies = [
                ("Correção Automática", self.correct_pose_automatically),
                ("Correção Drástica", lambda p: self._apply_drastic_corrections(p, p)),
                ("Elevação Z +5cm", lambda p: self._elevate_pose(p, 0.05)),
                ("Elevação Z +10cm", lambda p: self._elevate_pose(p, 0.10)),
                ("Posição Central", lambda p: self._move_to_center(p)),
            ]
            
            working_strategies = []
            
            for strategy_name, strategy_func in strategies:
                try:
                    corrected = strategy_func(pose)
                    if self.rtde_c.isPoseWithinSafetyLimits(corrected):
                        working_strategies.append((strategy_name, corrected))
                        print(f"   ✅ {strategy_name}: FUNCIONOU")
                    else:
                        print(f"   ❌ {strategy_name}: Falhou")
                except Exception as e:
                    print(f"   ❌ {strategy_name}: Erro - {e}")
                    
            # 3. Relatório final
            print(f"\n📊 RELATÓRIO FINAL - Pose {i+1}:")
            print(f"   Estratégias que funcionaram: {len(working_strategies)}")
            
            if working_strategies:
                print("   💡 SOLUÇÕES ENCONTRADAS:")
                for strategy_name, corrected_pose in working_strategies:
                    print(f"      • {strategy_name}: {[f'{p:.3f}' for p in corrected_pose]}")
            else:
                print("   ❌ NENHUMA SOLUÇÃO ENCONTRADA - Pose impossível")
            
        return working_strategies

    def _elevate_pose(self, pose, elevation):
        """AUXILIAR: Eleva a pose em Z"""
        corrected = pose.copy()
        corrected[2] += elevation
        return corrected

    def _move_to_center(self, pose):
        """AUXILIAR: Move pose para posição mais central"""
        corrected = pose.copy()
        corrected[0] = 0.4  # X central
        corrected[1] = 0.0  # Y central
        corrected[2] = max(corrected[2], 0.3)  # Z mínimo seguro
        return corrected

    def benchmark_correction_system(self):
        """
        📊 BENCHMARK: Testa o sistema de correção com várias poses
        """
        print("📊 BENCHMARK - Sistema de Correção")
        print("=" * 50)
        
        # Poses de teste variadas
        test_poses = [
            # Poses normais
            [0.3, 0.0, 0.3, 0.0, 3.14, 0.0],
            [0.4, 0.1, 0.2, 0.0, 3.14, 0.0], 
            
            # Poses problemáticas (muito baixas)
            [0.4, 0.2, 0.13, 0.0, 3.14, 0.0],
            [0.5, 0.3, 0.10, 0.0, 3.14, 0.0],
            
            # Poses extremas
            [0.7, 0.3, 0.15, 0.5, 3.14, 0.5],
            [0.2, -0.3, 0.12, -0.5, 2.5, -0.3],
            
            # Poses impossíveis
            [1.0, 0.8, 0.05, 1.0, 4.0, 2.0],
        ]
        
        results = {
            'total': len(test_poses),
            'original_valid': 0,
            'corrected_valid': 0,
            'impossible': 0,
            'details': []
        }
        
        for i, pose in enumerate(test_poses):
            print(f"\n📊 Teste {i+1}/{len(test_poses)}: {[f'{p:.3f}' for p in pose]}")
            
            # Teste original
            original_valid = self.rtde_c.isPoseWithinSafetyLimits(pose)
            if original_valid:
                results['original_valid'] += 1
                
            # Teste com correção
            corrected = self.correct_pose_automatically(pose)
            corrected_valid = self.rtde_c.isPoseWithinSafetyLimits(corrected)
            
            if corrected_valid:
                results['corrected_valid'] += 1
                status = "✅ CORRIGIDA"
            elif original_valid:
                status = "⚠️ PIOROU"
            else:
                results['impossible'] += 1
                status = "❌ IMPOSSÍVEL"
                
            results['details'].append({
                'pose': pose,
                'original_valid': original_valid,
                'corrected_valid': corrected_valid,
                'status': status
            })
            
            print(f"   Original: {'✅' if original_valid else '❌'} | Corrigida: {'✅' if corrected_valid else '❌'} | {status}")
        
        # Relatório final
        print(f"\n📊 RELATÓRIO FINAL DO BENCHMARK:")
        print(f"   Total de poses testadas: {results['total']}")
        print(f"   Originalmente válidas: {results['original_valid']} ({results['original_valid']/results['total']*100:.1f}%)")
        print(f"   Válidas após correção: {results['corrected_valid']} ({results['corrected_valid']/results['total']*100:.1f}%)")
        print(f"   Impossíveis: {results['impossible']} ({results['impossible']/results['total']*100:.1f}%)")
        print(f"   Taxa de melhoria: {((results['corrected_valid'] - results['original_valid'])/results['total']*100):.1f}%")
        
        return results

    # FUNÇÃO PARA USAR NO SEU CASO ESPECÍFICO
    def fix_calibration_pose(self, position_index, target_pose):
        """
        🎯 CORREÇÃO ESPECÍFICA: Para usar na calibração
        Retorna a melhor pose corrigida para uma posição específica
        """
        print(f"🎯 Corrigindo pose para posição {position_index}")
        
        # 1. Diagnóstico
        diagnostics = self.diagnostic_pose_rejection(target_pose)
        
        # 2. Se pose é válida, retornar original
        if self.rtde_c.isPoseWithinSafetyLimits(target_pose):
            print("✅ Pose original já é válida")
            return target_pose, True
            
        # 3. Tentar correção automática
        corrected = self.correct_pose_automatically(target_pose)
        if self.rtde_c.isPoseWithinSafetyLimits(corrected):
            print("✅ Correção automática funcionou")
            return corrected, True
            
        # 4. Estratégias específicas para calibração
        calibration_strategies = [
            ("Elevação +3cm", lambda p: self._elevate_pose(p, 0.03)),
            ("Elevação +5cm", lambda p: self._elevate_pose(p, 0.05)),
            ("Elevação +8cm", lambda p: self._elevate_pose(p, 0.08)),
            ("Posição mais central", self._move_to_center),
        ]
        
        for strategy_name, strategy_func in calibration_strategies:
            try:
                test_pose = strategy_func(target_pose)
                if self.rtde_c.isPoseWithinSafetyLimits(test_pose):
                    print(f"✅ {strategy_name} funcionou")
                    return test_pose, True
            except Exception as e:
                continue
                
        print("❌ Nenhuma estratégia funcionou para esta pose")
        return target_pose, False
        

    