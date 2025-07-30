"""
Main de Teste - Tapatan Robótico
Interface simples para testar movimentação do robô e lógica do jogo
"""

import os
import sys
from typing import Optional
from services.game_orchestrator import TapatanOrchestrator, ConfiguracaoOrquestrador


class TapatanTestInterface:
    """Interface de teste simples para o Tapatan"""
    
    def __init__(self):
        self.orquestrador: Optional[TapatanOrchestrator] = None
        self.config = ConfiguracaoOrquestrador(
            profundidade_ia=3,  # IA mais rápida para testes
            debug_mode=True,
            pausa_entre_jogadas=1.0,  # Pausa menor para testes
            velocidade_normal=0.05,   # Movimento mais lento para visualizar
            auto_calibrar=False       # Não calibrar automaticamente
        )
    
    def mostrar_banner(self):
        """Mostra banner inicial"""
        print("=" * 60)
        print("🎮 TAPATAN ROBÓTICO - MODO TESTE 🤖")
        print("=" * 60)
        print("Testando movimentação do robô e lógica do jogo")
        print("Robô apenas se posiciona sobre as casas (sem garra)")
        print("=" * 60)
        print()
    
    def mostrar_tabuleiro(self, estado_jogo):
        """Mostra o tabuleiro atual de forma visual"""
        tabuleiro = estado_jogo['tabuleiro']
        
        # Mapeamento dos valores
        simbolos = {0: ' ', 1: '🤖', 2: '👤'}
        
        print("\n" + "="*30)
        print("    TABULEIRO TAPATAN")
        print("="*30)
        print(f"  {simbolos[tabuleiro[0]]} | {simbolos[tabuleiro[1]]} | {simbolos[tabuleiro[2]]}")
        print("  ----+-----+----")
        print(f"  {simbolos[tabuleiro[3]]} | {simbolos[tabuleiro[4]]} | {simbolos[tabuleiro[5]]}")
        print("  ----+-----+----")
        print(f"  {simbolos[tabuleiro[6]]} | {simbolos[tabuleiro[7]]} | {simbolos[tabuleiro[8]]}")
        print("="*30)
        
        # Mostrar numeração das posições
        print("\nNumeração das posições:")
        print("  0 | 1 | 2")
        print("  --+---+--")
        print("  3 | 4 | 5")
        print("  --+---+--")
        print("  6 | 7 | 8")
        print()
    
    def mostrar_info_jogo(self, estado_jogo):
        """Mostra informações do jogo"""
        jogador_atual = "🤖 Robô" if estado_jogo['jogador_atual'] == 1 else "👤 Humano"
        fase = "Colocação" if estado_jogo['fase'] == "colocacao" else "Movimento"
        
        print(f"👾 Jogador atual: {jogador_atual}")
        print(f"⚡ Fase: {fase}")
        print(f"🤖 Peças robô: {estado_jogo['pecas_colocadas'][1]}/3")
        print(f"👤 Peças humano: {estado_jogo['pecas_colocadas'][2]}/3")
        
        if estado_jogo['jogo_terminado']:
                vencedor = "🤖 Robô" if estado_jogo['vencedor'] == 1 else "👤 Humano"
                print(f"🏆 VENCEDOR: {vencedor}!")
        print()
    
    def obter_jogada_humano(self, estado_jogo):
        """Obtém jogada do jogador humano via terminal"""
        try:
            if estado_jogo['fase'] == "colocacao":
                print("🎯 Sua vez! Escolha uma posição para colocar sua peça (0-8):")
                while True:
                    try:
                        posicao = int(input("Digite a posição: "))
                        if 0 <= posicao <= 8:
                            if posicao in estado_jogo['movimentos_validos']:
                                return {'posicao': posicao}
                            else:
                                print("❌ Posição já ocupada! Tente outra.")
                        else:
                            print("❌ Posição inválida! Use números de 0 a 8.")
                    except ValueError:
                        print("❌ Digite apenas números!")
            
            else:  # fase de movimento
                print("🎯 Sua vez! Escolha origem e destino para mover sua peça:")
                print("Suas peças estão nas posições:", end=" ")
                tabuleiro = estado_jogo['tabuleiro']
                pecas_humano = [i for i, v in enumerate(tabuleiro) if v == 2]
                print(pecas_humano)
                
                while True:
                    try:
                        origem = int(input("Digite a posição de origem: "))
                        if origem not in pecas_humano:
                            print("❌ Você não tem peça nesta posição!")
                            continue
                            
                        destino = int(input("Digite a posição de destino: "))
                        if 0 <= destino <= 8:
                            # Verificar se é movimento válido
                            movimentos_validos = estado_jogo['movimentos_validos']
                            movimento_valido = any(mov[0] == origem and mov[1] == destino 
                                                 for mov in movimentos_validos)
                            if movimento_valido:
                                return {'origem': origem, 'destino': destino}
                            else:
                                print("❌ Movimento inválido! Só pode mover para posições adjacentes vazias.")
                        else:
                            print("❌ Posição inválida! Use números de 0 a 8.")
                    except ValueError:
                        print("❌ Digite apenas números!")
                        
        except KeyboardInterrupt:
            print("\n\n👋 Saindo do jogo...")
            return None
    
    def aguardar_confirmacao_robo(self):
        """Aguarda confirmação de que o robô executou o movimento"""
        print("🤖 Robô está executando movimento...")
        input("⏳ Pressione ENTER após o robô completar o movimento...")
    
    def menu_principal(self):
        """Menu principal da interface"""
        while True:
            print("\n" + "="*40)
            print("MENU PRINCIPAL")
            print("="*40)
            print("1. 🚀 Iniciar nova partida")
            print("2. 🔧 Calibrar sistema")
            print("3. 📊 Ver status do sistema")
            print("4. 🚨 Parada de emergência")
            print("5. 👋 Sair")
            print("="*40)
            
            try:
                opcao = input("Escolha uma opção: ").strip()
                
                if opcao == "1":
                    self.executar_partida()
                elif opcao == "2":
                    self.calibrar_sistema()
                elif opcao == "3":
                    self.mostrar_status()
                elif opcao == "4":
                    self.parada_emergencia()
                elif opcao == "5":
                    print("👋 Até logo!")
                    break
                else:
                    print("❌ Opção inválida!")
                    
            except KeyboardInterrupt:
                print("\n👋 Saindo...")
                break
    
    def executar_partida(self):
        """Executa uma partida completa"""
        if not self.orquestrador:
            print("❌ Sistema não inicializado!")
            return
            
        print("\n🎮 Iniciando nova partida...")
        
        if not self.orquestrador.iniciar_partida():
            print("❌ Erro ao iniciar partida!")
            return
        
        # Loop principal do jogo
        while True:
            try:
                estado_jogo = self.orquestrador.game_service.obter_estado_jogo()
                
                # Mostrar estado atual
                self.mostrar_tabuleiro(estado_jogo)
                self.mostrar_info_jogo(estado_jogo)
                
                # Verificar se jogo terminou
                if estado_jogo['jogo_terminado']:
                    print("🎮 Jogo terminado!")
                    input("Pressione ENTER para continuar...")
                    break
                
                # Vez do humano
                if estado_jogo['jogador_atual'] == 2:
                    jogada = self.obter_jogada_humano(estado_jogo)
                    if jogada is None:  # Usuário cancelou
                        break
                        
                    # Processar jogada
                    if 'posicao' in jogada:
                        resultado = self.orquestrador.processar_jogada_humano(posicao=jogada['posicao'])
                    else:
                        resultado = self.orquestrador.processar_jogada_humano(
                            origem=jogada['origem'], destino=jogada['destino'])
                    
                    if not resultado['sucesso']:
                        print(f"❌ Erro: {resultado['mensagem']}")
                        continue
                        
                    print("✅ Sua jogada foi executada!")
                    
                    # Se há jogada do robô na resposta, mostrar
                    if 'jogada_robo' in resultado:
                        jogada_robo = resultado['jogada_robo']['jogada']
                        if 'posicao' in jogada_robo:
                            print(f"🤖 Robô colocou peça na posição {jogada_robo['posicao']}")
                        else:
                            print(f"🤖 Robô moveu peça de {jogada_robo['origem']} para {jogada_robo['destino']}")
                        
                        # Aguardar confirmação do movimento físico
                        self.aguardar_confirmacao_robo()
                
                # Vez do robô (só acontece se não houve jogada automática)
                elif estado_jogo['jogador_atual'] == 1:
                    input("🤖 Vez do robô. Pressione ENTER para continuar...")
                    
                    resultado = self.orquestrador.executar_jogada_robo()
                    
                    if resultado['sucesso']:
                        jogada = resultado['jogada']
                        if 'posicao' in jogada:
                            print(f"🤖 Robô colocou peça na posição {jogada['posicao']}")
                        else:
                            print(f"🤖 Robô moveu peça de {jogada['origem']} para {jogada['destino']}")
                        
                        # Aguardar confirmação do movimento físico
                        self.aguardar_confirmacao_robo()
                    else:
                        print(f"❌ Erro na jogada do robô: {resultado['mensagem']}")
                        break
                
            except KeyboardInterrupt:
                print("\n\n🛑 Partida interrompida pelo usuário!")
                break
            except Exception as e:
                print(f"❌ Erro durante a partida: {e}")
                break
    
    def calibrar_sistema(self):
        """Executa calibração do sistema"""
        if not self.orquestrador:
            print("❌ Sistema não inicializado!")
            return
            
        print("\n🔧 Iniciando calibração do sistema...")
        print("⚠️  O robô vai visitar algumas posições do tabuleiro.")
        
        if input("Continuar? (s/N): ").lower().startswith('s'):
            if self.orquestrador.calibrar_sistema():
                print("✅ Calibração concluída com sucesso!")
            else:
                print("❌ Falha na calibração!")
        else:
            print("Calibração cancelada.")
    
    def mostrar_status(self):
        """Mostra status completo do sistema"""
        if not self.orquestrador:
            print("❌ Sistema não inicializado!")
            return
            
        print("\n📊 STATUS DO SISTEMA")
        print("="*40)
        
        status = self.orquestrador.obter_status_completo()
        
        # Status do orquestrador
        print(f"🎮 Orquestrador: {status['orquestrador']['status']}")
        print(f"🎯 Jogo ativo: {status['orquestrador']['jogo_ativo']}")
        
        # Status do robô
        if status['robot']:
            robot_status = status['robot']
            print(f"🤖 Robô: {'Conectado' if robot_status['connected'] else 'Desconectado'}")
            if robot_status['current_pose']:
                pose = robot_status['current_pose']
                print(f"📍 Posição: X={pose['x']:.3f}, Y={pose['y']:.3f}, Z={pose['z']:.3f}")
        
        # Status do jogo
        if status['jogo']:
            jogo = status['jogo']
            print(f"⚡ Fase: {jogo['fase']}")
            print(f"👾 Jogador atual: {jogo['jogador_atual']}")
        
        # Erros
        if status['orquestrador']['ultimo_erro']:
            print(f"❌ Último erro: {status['orquestrador']['ultimo_erro']}")
        
        print("="*40)
        input("Pressione ENTER para continuar...")
    
    def parada_emergencia(self):
        """Executa parada de emergência"""
        if not self.orquestrador:
            print("❌ Sistema não inicializado!")
            return
            
        print("\n🚨 PARADA DE EMERGÊNCIA")
        if input("⚠️  Confirma parada de emergência? (s/N): ").lower().startswith('s'):
            if self.orquestrador.parada_emergencia():
                print("🚨 PARADA DE EMERGÊNCIA EXECUTADA!")
            else:
                print("❌ Falha ao executar parada de emergência!")
    
    def inicializar_sistema(self):
        """Inicializa o sistema completo"""
        print("🚀 Inicializando sistema Tapatan...")
        
        try:
            self.orquestrador = TapatanOrchestrator(self.config)
            
            if self.orquestrador.inicializar():
                print("✅ Sistema inicializado com sucesso!")
                return True
            else:
                print("❌ Falha na inicialização do sistema!")
                return False
                
        except Exception as e:
            print(f"❌ Erro na inicialização: {e}")
            return False
    
    def finalizar_sistema(self):
        """Finaliza o sistema"""
        if self.orquestrador:
            print("🔚 Finalizando sistema...")
            self.orquestrador.finalizar()
            print("✅ Sistema finalizado!")
    
    def executar(self):
        """Execução principal da interface"""
        self.mostrar_banner()
        
        # Inicializar sistema
        if not self.inicializar_sistema():
            print("❌ Não foi possível inicializar o sistema!")
            return
        
        try:
            # Menu principal
            self.menu_principal()
        except Exception as e:
            print(f"❌ Erro durante execução: {e}")
        finally:
            # Finalizar sistema
            self.finalizar_sistema()


def main():
    """Função principal"""
    try:
        interface = TapatanTestInterface()
        interface.executar()
    except KeyboardInterrupt:
        print("\n\n👋 Programa interrompido pelo usuário!")
    except Exception as e:
        print(f"❌ Erro fatal: {e}")
    
    print("\n🔚 Programa finalizado.")


if __name__ == "__main__":
    main()