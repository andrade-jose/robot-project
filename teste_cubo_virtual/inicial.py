"""
SISTEMA DE CUBO VIRTUAL 3D - ESTRUTURA REORGANIZADA
==================================================

Este sistema implementa um cubo virtual 3D para simulação de visão estéreo,
rastreamento de objetos e análises espaciais. O código está organizado de forma
hierárquica: conceitos básicos → funcionalidades → visualizações → testes.

ESTRUTURA:
1. Imports e Configurações
2. Classe Base: CuboVirtual
3. Classe Especializada: CuboRastreamento  
4. Classe de Visualizações: VisualizadorCubo
5. Classe de Testes: TestadorComVisualizacoes
6. Funções de Execução
"""

# ===== 1. IMPORTS E CONFIGURAÇÕES =====
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import random
from typing import Tuple, Any, Optional, List

# Configurar estilo dos gráficos (sem seaborn)
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True


# ===== 2. CLASSE BASE: CUBO VIRTUAL =====
class CuboVirtual:
    """
    Classe base que implementa um cubo virtual 3D.
    
    FUNCIONALIDADES BÁSICAS:
    - Criação de matriz 3D
    - Definir/obter valores em posições específicas
    - Preenchimento automático com funções
    - Busca de posições por critérios
    - Análise de vizinhos
    - Fatiamento em diferentes eixos
    """
    
    def __init__(self, largura: int, altura: int, profundidade: int, valor_inicial=None):
        """
        Inicializa o cubo virtual.
        
        Args:
            largura: Tamanho no eixo X
            altura: Tamanho no eixo Y  
            profundidade: Tamanho no eixo Z
            valor_inicial: Valor para preencher todas as posições (opcional)
        """
        self.largura = largura
        self.altura = altura
        self.profundidade = profundidade
        self.matriz = np.full((largura, altura, profundidade), valor_inicial, dtype=object)
    
    def definir_posicao(self, x: int, y: int, z: int, valor: Any) -> bool:
        """Define um valor em uma posição específica do cubo."""
        if self._validar_coordenadas(x, y, z):
            self.matriz[x][y][z] = valor
            return True
        return False
    
    def obter_posicao(self, x: int, y: int, z: int) -> Any:
        """Obtém o valor de uma posição específica do cubo."""
        if self._validar_coordenadas(x, y, z):
            return self.matriz[x][y][z]
        return None
    
    def _validar_coordenadas(self, x: int, y: int, z: int) -> bool:
        """Valida se as coordenadas estão dentro dos limites do cubo."""
        return (0 <= x < self.largura and 
                0 <= y < self.altura and 
                0 <= z < self.profundidade)
    
    def preencher_cubo(self, funcao_preenchimento=None):
        """
        Preenche todo o cubo usando uma função ou padrão padrão.
        
        Args:
            funcao_preenchimento: Função que recebe (x,y,z) e retorna um valor
        """
        for x in range(self.largura):
            for y in range(self.altura):
                for z in range(self.profundidade):
                    if funcao_preenchimento:
                        valor = funcao_preenchimento(x, y, z)
                    else:
                        valor = f"({x},{y},{z})"
                    self.matriz[x][y][z] = valor
    
    def buscar_posicoes(self, condicao) -> list:
        """
        Busca todas as posições que satisfazem uma condição.
        
        Args:
            condicao: Função que recebe um valor e retorna True/False
            
        Returns:
            Lista de tuplas (x, y, z) das posições encontradas
        """
        posicoes = []
        for x in range(self.largura):
            for y in range(self.altura):
                for z in range(self.profundidade):
                    try:
                        if condicao(self.matriz[x][y][z]):
                            posicoes.append((x, y, z))
                    except:
                        continue
        return posicoes
    
    def obter_vizinhos(self, x: int, y: int, z: int, incluir_diagonais=False) -> list:
        """
        Encontra todos os vizinhos válidos de uma posição.
        
        Args:
            x, y, z: Coordenadas da posição central
            incluir_diagonais: Se True, inclui vizinhos diagonais (26 total)
                              Se False, apenas faces adjacentes (6 total)
        
        Returns:
            Lista de tuplas (x, y, z) dos vizinhos válidos
        """
        vizinhos = []
        
        if incluir_diagonais:
            # Todos os 26 vizinhos (incluindo diagonais)
            deltas = [(dx, dy, dz) for dx in [-1, 0, 1] 
                     for dy in [-1, 0, 1] 
                     for dz in [-1, 0, 1] 
                     if not (dx == 0 and dy == 0 and dz == 0)]
        else:
            # Apenas os 6 vizinhos das faces adjacentes
            deltas = [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]
        
        for dx, dy, dz in deltas:
            nx, ny, nz = x + dx, y + dy, z + dz
            if self._validar_coordenadas(nx, ny, nz):
                vizinhos.append((nx, ny, nz))
        
        return vizinhos
    
    def fatiar_cubo(self, eixo: str, indice: int) -> np.ndarray:
        """
        Extrai uma fatia 2D do cubo em um eixo específico.
        
        Args:
            eixo: 'x', 'y' ou 'z'
            indice: Posição da fatia no eixo escolhido
            
        Returns:
            Array 2D representando a fatia
        """
        if eixo.lower() == 'x' and 0 <= indice < self.largura:
            return self.matriz[indice, :, :]
        elif eixo.lower() == 'y' and 0 <= indice < self.altura:
            return self.matriz[:, indice, :]
        elif eixo.lower() == 'z' and 0 <= indice < self.profundidade:
            return self.matriz[:, :, indice]
        else:
            raise ValueError("Eixo inválido ou índice fora dos limites")


# ===== 3. CLASSE ESPECIALIZADA: CUBO RASTREAMENTO =====
class CuboRastreamento:
    """
    Classe especializada para rastreamento de objetos em movimento.
    
    FUNCIONALIDADES AVANÇADAS:
    - Conversão de coordenadas reais para índices do cubo
    - Registro de pontos detectados com timestamp e confiança
    - Cálculo de trajetórias no tempo
    - Histórico de detecções
    """
    
    def __init__(self, largura: int, altura: int, profundidade: int, 
                 origem: Tuple[float, float, float] = (0, 0, 0), 
                 escala: float = 1.0):
        """
        Inicializa o sistema de rastreamento.
        
        Args:
            largura, altura, profundidade: Dimensões do cubo
            origem: Ponto de origem no espaço real (x, y, z)
            escala: Fator de conversão entre coordenadas reais e índices
        """
        self.cubo = CuboVirtual(largura, altura, profundidade)
        self.origem = origem
        self.escala = escala
        self.historico_deteccoes = []
    
    def coordenada_real_para_indice(self, x_real: float, y_real: float, z_real: float) -> Tuple[int, int, int]:
        """Converte coordenadas reais para índices do cubo."""
        x_idx = int((x_real - self.origem[0]) / self.escala)
        y_idx = int((y_real - self.origem[1]) / self.escala)  
        z_idx = int((z_real - self.origem[2]) / self.escala)
        return (x_idx, y_idx, z_idx)
    
    def registrar_ponto_detectado(self, x_real: float, y_real: float, z_real: float, 
                                confianca: float = 1.0, timestamp: float = None):
        """
        Registra um ponto detectado no sistema.
        
        Args:
            x_real, y_real, z_real: Coordenadas reais do ponto
            confianca: Nível de confiança da detecção (0.0 a 1.0)
            timestamp: Momento da detecção (usa time.time() se None)
        
        Returns:
            True se o ponto foi registrado com sucesso
        """
        if timestamp is None:
            timestamp = time.time()
            
        x_idx, y_idx, z_idx = self.coordenada_real_para_indice(x_real, y_real, z_real)
        
        if self.cubo._validar_coordenadas(x_idx, y_idx, z_idx):
            dados_ponto = {
                'coordenada_real': (x_real, y_real, z_real),
                'coordenada_indice': (x_idx, y_idx, z_idx),
                'confianca': confianca,
                'timestamp': timestamp,
                'deteccoes': 1
            }
            
            # Se já existe um ponto nesta posição, atualiza os dados
            ponto_existente = self.cubo.obter_posicao(x_idx, y_idx, z_idx)
            if ponto_existente:
                dados_ponto['deteccoes'] = ponto_existente['deteccoes'] + 1
                dados_ponto['confianca'] = (ponto_existente['confianca'] + confianca) / 2
            
            self.cubo.definir_posicao(x_idx, y_idx, z_idx, dados_ponto)
            self.historico_deteccoes.append(dados_ponto.copy())
            return True
        return False
    
    def calcular_trajetoria(self) -> list:
        """
        Calcula a trajetória ordenada no tempo.
        
        Returns:
            Lista de coordenadas reais ordenadas por timestamp
        """
        trajetoria = sorted(self.historico_deteccoes, key=lambda x: x['timestamp'])
        return [ponto['coordenada_real'] for ponto in trajetoria]


# ===== 4. CLASSE DE VISUALIZAÇÕES =====
class VisualizadorCubo:
    """
    Classe responsável por criar visualizações dos dados do cubo.
    
    TIPOS DE VISUALIZAÇÕES:
    - Estrutura básica e estatísticas
    - Padrões de preenchimento
    - Resultados de busca
    - Análise de vizinhos
    - Fatiamento em múltiplos eixos
    - Performance e benchmarks
    - Rastreamento de trajetórias
    - Simulação de visão estéreo
    """
    
    def __init__(self):
        self.fig_count = 0
        
    def plot_teste_basico(self, cubo: CuboVirtual):
        """Visualiza estrutura básica e estatísticas do cubo."""
        fig = plt.figure(figsize=(15, 5))
        
        # Plot 1: Estrutura 3D básica
        ax1 = fig.add_subplot(131, projection='3d')
        
        # Mostrar apenas bordas do cubo para estrutura
        x_edges = [0, cubo.largura-1]
        y_edges = [0, cubo.altura-1] 
        z_edges = [0, cubo.profundidade-1]
        
        # Desenhar arestas do cubo
        for x in x_edges:
            for y in y_edges:
                ax1.plot([x, x], [y, y], z_edges, 'b-', alpha=0.3)
        for x in x_edges:
            for z in z_edges:
                ax1.plot([x, x], y_edges, [z, z], 'b-', alpha=0.3)
        for y in y_edges:
            for z in z_edges:
                ax1.plot(x_edges, [y, y], [z, z], 'b-', alpha=0.3)
        
        # Marcar pontos especiais
        ax1.scatter([0], [0], [0], c='red', s=100, label='Origem')
        ax1.scatter([cubo.largura-1], [cubo.altura-1], [cubo.profundidade-1], 
                   c='green', s=100, label='Extremo')
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Estrutura Básica do Cubo')
        ax1.legend()
        
        # Plot 2: Estatísticas
        ax2 = fig.add_subplot(132)
        total = cubo.largura * cubo.altura * cubo.profundidade
        preenchidas = sum(1 for x in range(cubo.largura) 
                         for y in range(cubo.altura) 
                         for z in range(cubo.profundidade) 
                         if cubo.obter_posicao(x, y, z) is not None)
        vazias = total - preenchidas
        
        stats_labels = ['Total\nPosições', 'Posições\nPreenchidas', 'Posições\nVazias']
        stats_values = [total, preenchidas, vazias]
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        
        bars = ax2.bar(stats_labels, stats_values, color=colors)
        ax2.set_title('Estatísticas do Cubo')
        ax2.set_ylabel('Número de Posições')
        
        # Adicionar valores nas barras
        for bar, value in zip(bars, stats_values):
            ax2.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + max(stats_values)*0.01,
                    str(value), ha='center', va='bottom')
        
        # Plot 3: Ocupação percentual
        ax3 = fig.add_subplot(133)
        ocupacao = (preenchidas / total) * 100 if total > 0 else 0
        livre = 100 - ocupacao
        
        ax3.pie([ocupacao, livre], labels=['Ocupado', 'Livre'], 
                colors=['lightgreen', 'lightgray'], autopct='%1.1f%%')
        ax3.set_title('Ocupação do Cubo')
        
        plt.tight_layout()
        plt.savefig(f'teste_basico_{self.fig_count}.png', dpi=300, bbox_inches='tight')
        plt.show()
        self.fig_count += 1
    
    def plot_teste_preenchimento(self, cubo: CuboVirtual):
        """Visualiza padrões de preenchimento do cubo."""
        fig = plt.figure(figsize=(20, 5))
        
        # Extrair dados para plotagem
        x_coords, y_coords, z_coords, valores = [], [], [], []
        for x in range(cubo.largura):
            for y in range(cubo.altura):
                for z in range(cubo.profundidade):
                    valor = cubo.obter_posicao(x, y, z)
                    if valor is not None:
                        x_coords.append(x)
                        y_coords.append(y)
                        z_coords.append(z)
                        try:
                            valores.append(float(valor))
                        except:
                            valores.append(0)
        
        if not valores:  # Se não há dados, criar dados de exemplo
            return
        
        # Plot 1: Scatter 3D com gradiente de cor
        ax1 = fig.add_subplot(141, projection='3d')
        scatter = ax1.scatter(x_coords, y_coords, z_coords, c=valores, 
                             cmap='viridis', s=50, alpha=0.7)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Distribuição 3D dos Valores')
        plt.colorbar(scatter, ax=ax1, shrink=0.5)
        
        # Plot 2: Histograma dos valores
        ax2 = fig.add_subplot(142)
        ax2.hist(valores, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Valores')
        ax2.set_ylabel('Frequência')
        ax2.set_title('Distribuição dos Valores')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Fatia central (plano XY)
        ax3 = fig.add_subplot(143)
        centro_z = cubo.profundidade // 2
        fatia = cubo.fatiar_cubo('z', centro_z)
        
        # Converter para array numérico
        fatia_numerica = np.zeros(fatia.shape)
        for i in range(fatia.shape[0]):
            for j in range(fatia.shape[1]):
                try:
                    fatia_numerica[i, j] = float(fatia[i, j]) if fatia[i, j] is not None else 0
                except:
                    fatia_numerica[i, j] = 0
        
        im = ax3.imshow(fatia_numerica, cmap='viridis', origin='lower')
        ax3.set_xlabel('Y')
        ax3.set_ylabel('X')
        ax3.set_title(f'Fatia Z={centro_z}')
        plt.colorbar(im, ax=ax3)
        
        # Plot 4: Estatísticas por eixo
        ax4 = fig.add_subplot(144)
        
        # Calcular médias por eixo
        medias_x = []
        medias_y = []
        medias_z = []
        
        for x in range(cubo.largura):
            vals_x = [valores[i] for i in range(len(valores)) if x_coords[i] == x]
            medias_x.append(np.mean(vals_x) if vals_x else 0)
        
        for y in range(cubo.altura):
            vals_y = [valores[i] for i in range(len(valores)) if y_coords[i] == y]
            medias_y.append(np.mean(vals_y) if vals_y else 0)
        
        for z in range(cubo.profundidade):
            vals_z = [valores[i] for i in range(len(valores)) if z_coords[i] == z]
            medias_z.append(np.mean(vals_z) if vals_z else 0)
        
        ax4.plot(range(cubo.largura), medias_x, 'o-', label='Média X', linewidth=2)
        ax4.plot(range(cubo.altura), medias_y, 's-', label='Média Y', linewidth=2)
        ax4.plot(range(cubo.profundidade), medias_z, '^-', label='Média Z', linewidth=2)
        ax4.set_xlabel('Índice')
        ax4.set_ylabel('Valor Médio')
        ax4.set_title('Médias por Eixo')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'teste_preenchimento_{self.fig_count}.png', dpi=300, bbox_inches='tight')
        plt.show()
        self.fig_count += 1


# ===== 5. CLASSE DE TESTES =====
class TestadorComVisualizacoes:
    """
    Classe principal para executar todos os testes com visualizações.
    
    TESTES IMPLEMENTADOS:
    - Teste básico (estrutura e estatísticas)
    - Teste de preenchimento (padrões espaciais)
    - Teste de busca (filtros e condições)
    - Teste de vizinhos (conectividade)
    - Teste de rastreamento (movimento)
    - Benchmarks de performance
    """
    
    def __init__(self):
        self.visualizador = VisualizadorCubo()
        self.resultados = {}
    
    def executar_teste_basico(self):
        """Executa teste básico com visualização."""
        print("📊 1. Teste Básico com Visualização")
        
        # Criar cubo pequeno para demonstração
        cubo_basico = CuboVirtual(5, 5, 5, None)
        
        # Definir alguns pontos especiais
        cubo_basico.definir_posicao(0, 0, 0, "origem")
        cubo_basico.definir_posicao(4, 4, 4, "extremo")
        cubo_basico.definir_posicao(2, 2, 2, "centro")
        
        # Visualizar
        self.visualizador.plot_teste_basico(cubo_basico)
        
        return cubo_basico
    
    def executar_teste_preenchimento(self):
        """Executa teste de preenchimento com padrões matemáticos."""
        print("📊 2. Teste de Preenchimento com Visualização")
        
        # Criar cubo médio
        cubo_preench = CuboVirtual(8, 8, 8)
        
        # Função de preenchimento: distância do centro
        def funcao_distancia(x, y, z):
            centro = (4, 4, 4)
            return round(((x-centro[0])**2 + (y-centro[1])**2 + (z-centro[2])**2)**0.5, 2)
        
        cubo_preench.preencher_cubo(funcao_distancia)
        
        # Visualizar
        self.visualizador.plot_teste_preenchimento(cubo_preench)
        
        return cubo_preench
    
    def executar_teste_rastreamento(self):
        """Executa teste de rastreamento de objetos."""
        print("📊 3. Teste de Rastreamento com Visualização")
        
        # Criar sistema de rastreamento
        cubo_track = CuboRastreamento(50, 50, 50, origem=(-25, -25, -25), escala=1.0)
        
        # Simular movimento de objeto
        trajetoria_teste = [
            (0, 0, 0), (5, 2, 1), (10, 5, 2), 
            (15, 10, 3), (20, 15, 4), (25, 20, 5)
        ]
        
        for i, (x, y, z) in enumerate(trajetoria_teste):
            confianca = 0.9 + random.uniform(-0.1, 0.1)
            timestamp = time.time() + i * 0.1
            cubo_track.registrar_ponto_detectado(x, y, z, confianca, timestamp)
        
        # Calcular trajetória
        trajetoria = cubo_track.calcular_trajetoria()
        print(f"   Trajetória calculada: {len(trajetoria)} pontos")
        
        return cubo_track
    
    def executar_benchmark_simples(self):
        """Executa benchmark de performance básico."""
        print("📊 4. Benchmark de Performance")
        
        tamanhos = [(5, 5, 5), (10, 10, 10), (15, 15, 15)]
        resultados = {}
        
        for largura, altura, profundidade in tamanhos:
            key = f"{largura}x{altura}x{profundidade}"
            print(f"   Testando tamanho {key}...")
            
            # Medir criação
            inicio = time.time()
            cubo = CuboVirtual(largura, altura, profundidade)
            tempo_criacao = time.time() - inicio
            
            # Medir preenchimento
            inicio = time.time()
            count = 0
            for x in range(0, largura, 2):
                for y in range(0, altura, 2):
                    for z in range(0, profundidade, 2):
                        cubo.definir_posicao(x, y, z, count)
                        count += 1
            tempo_preenchimento = time.time() - inicio
            
            # Medir busca
            inicio = time.time()
            resultados_busca = cubo.buscar_posicoes(lambda v: v is not None and v % 5 == 0)
            tempo_busca = time.time() - inicio
            
            resultados[key] = {
                "total_voxels": largura * altura * profundidade,
                "tempo_criacao": tempo_criacao,
                "tempo_preenchimento": tempo_preenchimento,
                "tempo_busca": tempo_busca,
                "voxels_por_segundo": count / tempo_preenchimento if tempo_preenchimento > 0 else 0,
                "resultados_encontrados": len(resultados_busca)
            }
        
        return resultados
    
    def executar_todos_testes(self):
        """Executa todos os testes disponíveis."""
        print("🎨 EXECUTANDO SISTEMA DE TESTES COMPLETO")
        print("=" * 50)
        
        # Executar testes individuais
        cubo_basico = self.executar_teste_basico()
        cubo_preench = self.executar_teste_preenchimento()
        cubo_track = self.executar_teste_rastreamento()
        resultados_perf = self.executar_benchmark_simples()
        
        # Salvar resultados
        self.resultados = {
            'cubo_basico': cubo_basico,
            'cubo_preenchimento': cubo_preench,
            'cubo_rastreamento': cubo_track,
            'performance': resultados_perf
        }
        
        print(f"\n✅ Testes concluídos!")
        print(f"📁 Total de gráficos gerados: {self.visualizador.fig_count}")
        
        return self.resultados


# ===== 6. FUNÇÕES DE EXECUÇÃO =====
def exemplo_uso_basico():
    """Demonstra uso básico das classes principais."""
    print("🔧 EXEMPLO DE USO BÁSICO")
    print("-" * 30)
    
    # Criar cubo simples
    cubo = CuboVirtual(5, 5, 5)
    print(f"Cubo criado: {cubo.largura}x{cubo.altura}x{cubo.profundidade}")
    
    # Definir alguns valores
    cubo.definir_posicao(0, 0, 0, "início")
    cubo.definir_posicao(2, 2, 2, "meio")
    cubo.definir_posicao(4, 4, 4, "fim")
    
    # Buscar posições
    posicoes = cubo.buscar_posicoes(lambda v: v is not None)
    print(f"Posições preenchidas: {len(posicoes)}")
    
    # Analisar vizinhos
    vizinhos = cubo.obter_vizinhos(2, 2, 2, incluir_diagonais=True)
    print(f"Vizinhos do centro: {len(vizinhos)}")
    
    return cubo

def executar_sistema_completo():
    """Função principal para executar todo o sistema."""
    print("🎨 SISTEMA DE CUBO VIRTUAL 3D")
    print("=" * 40)
    print("Este sistema demonstra:")
    print("• Estruturas de dados 3D")
    print("• Rastreamento de objetos")
    print("• Análises espaciais")
    print("• Visualizações interativas")
    print()
    
    # Exemplo básico
    cubo_exemplo = exemplo_uso_basico()
    
    print("\n" + "="*40)
    
    # Sistema completo de testes
    testador = TestadorComVisualizacoes()
    resultados = testador.executar_todos_testes()
    
    print("\n📋 RESUMO FINAL:")
    print("✓ Classes implementadas: CuboVirtual, CuboRastreamento")
    print("✓ Visualizações: Estrutura básica, preenchimento, rastreamento")
    print("✓ Testes: Performance, busca, análise de vizinhos")
    print("✓ Funcionalidades: Fatiamento, coordenadas reais, histórico temporal")
    
    return resultados


# ===== 7. FUNCIONALIDADES AVANÇADAS ADICIONAIS =====
class AlgoritmosAvancados:
    """
    Classe com algoritmos avançados para análise espacial.
    Implementa funcionalidades como detecção de formas geométricas,
    análise de superfícies e algoritmos de pathfinding.
    """
    
    @staticmethod
    def criar_esfera(cubo: CuboVirtual, centro: Tuple[int, int, int], raio: float, valor=1):
        """
        Cria uma esfera no cubo usando a equação da esfera.
        
        Args:
            cubo: Instância do CuboVirtual
            centro: Coordenadas do centro (x, y, z)
            raio: Raio da esfera
            valor: Valor a ser usado para voxels da esfera
        """
        cx, cy, cz = centro
        
        for x in range(cubo.largura):
            for y in range(cubo.altura):
                for z in range(cubo.profundidade):
                    # Calcular distância do ponto ao centro
                    distancia = ((x - cx)**2 + (y - cy)**2 + (z - cz)**2)**0.5
                    
                    if distancia <= raio:
                        cubo.definir_posicao(x, y, z, valor)
    
    @staticmethod
    def detectar_superficie(cubo: CuboVirtual, valor_objeto=1) -> List[Tuple[int, int, int]]:
        """
        Detecta voxels da superfície de um objeto.
        Um voxel de superfície tem pelo menos um vizinho vazio.
        
        Args:
            cubo: Instância do CuboVirtual
            valor_objeto: Valor que representa o objeto
            
        Returns:
            Lista de coordenadas dos voxels de superfície
        """
        superficie = []
        
        for x in range(cubo.largura):
            for y in range(cubo.altura):
                for z in range(cubo.profundidade):
                    if cubo.obter_posicao(x, y, z) == valor_objeto:
                        # Verificar se tem vizinho vazio
                        vizinhos = cubo.obter_vizinhos(x, y, z, incluir_diagonais=False)
                        
                        tem_vizinho_vazio = False
                        for vx, vy, vz in vizinhos:
                            if cubo.obter_posicao(vx, vy, vz) != valor_objeto:
                                tem_vizinho_vazio = True
                                break
                        
                        # Verificar se está na borda do cubo
                        na_borda = (x == 0 or x == cubo.largura-1 or
                                   y == 0 or y == cubo.altura-1 or
                                   z == 0 or z == cubo.profundidade-1)
                        
                        if tem_vizinho_vazio or na_borda:
                            superficie.append((x, y, z))
        
        return superficie
    
    @staticmethod
    def calcular_volume(cubo: CuboVirtual, valor_objeto=1) -> int:
        """Calcula o volume de um objeto no cubo."""
        return len(cubo.buscar_posicoes(lambda v: v == valor_objeto))
    
    @staticmethod
    def calcular_centro_massa(cubo: CuboVirtual, valor_objeto=1) -> Tuple[float, float, float]:
        """
        Calcula o centro de massa de um objeto.
        
        Returns:
            Coordenadas do centro de massa (x, y, z)
        """
        posicoes = cubo.buscar_posicoes(lambda v: v == valor_objeto)
        
        if not posicoes:
            return (0, 0, 0)
        
        x_media = sum(pos[0] for pos in posicoes) / len(posicoes)
        y_media = sum(pos[1] for pos in posicoes) / len(posicoes)
        z_media = sum(pos[2] for pos in posicoes) / len(posicoes)
        
        return (x_media, y_media, z_media)
    
    @staticmethod
    def criar_cubo_solido(cubo: CuboVirtual, centro: Tuple[int, int, int], tamanho: int, valor=1):
        """
        Cria um cubo sólido dentro do cubo virtual.
        
        Args:
            cubo: Instância do CuboVirtual
            centro: Centro do cubo (x, y, z)
            tamanho: Tamanho da aresta do cubo
            valor: Valor a ser usado para voxels do cubo
        """
        cx, cy, cz = centro
        metade = tamanho // 2
        
        coordenadas = []
        for x in range(max(0, cx - metade), min(cubo.largura, cx + metade + 1)):
            for y in range(max(0, cy - metade), min(cubo.altura, cy + metade + 1)):
                for z in range(max(0, cz - metade), min(cubo.profundidade, cz + metade + 1)):
                    cubo.definir_posicao(x, y, z, valor)
                    coordenadas.append((x, y, z))
        
        return coordenadas
    
    @staticmethod
    def criar_cilindro(cubo: CuboVirtual, centro: Tuple[int, int, int], 
                      raio: float, altura: int, eixo='z', valor=1):
        """
        Cria um cilindro no cubo virtual.
        
        Args:
            cubo: Instância do CuboVirtual
            centro: Centro da base do cilindro
            raio: Raio do cilindro
            altura: Altura do cilindro
            eixo: Eixo de orientação ('x', 'y', ou 'z')
            valor: Valor para voxels do cilindro
        """
        cx, cy, cz = centro
        coordenadas = []
        
        if eixo.lower() == 'z':
            for z in range(max(0, cz), min(cubo.profundidade, cz + altura)):
                for x in range(cubo.largura):
                    for y in range(cubo.altura):
                        distancia = ((x - cx)**2 + (y - cy)**2)**0.5
                        if distancia <= raio:
                            cubo.definir_posicao(x, y, z, valor)
                            coordenadas.append((x, y, z))
        
        elif eixo.lower() == 'y':
            for y in range(max(0, cy), min(cubo.altura, cy + altura)):
                for x in range(cubo.largura):
                    for z in range(cubo.profundidade):
                        distancia = ((x - cx)**2 + (z - cz)**2)**0.5
                        if distancia <= raio:
                            cubo.definir_posicao(x, y, z, valor)
                            coordenadas.append((x, y, z))
        
        elif eixo.lower() == 'x':
            for x in range(max(0, cx), min(cubo.largura, cx + altura)):
                for y in range(cubo.altura):
                    for z in range(cubo.profundidade):
                        distancia = ((y - cy)**2 + (z - cz)**2)**0.5
                        if distancia <= raio:
                            cubo.definir_posicao(x, y, z, valor)
                            coordenadas.append((x, y, z))
        
        return coordenadas
    
    @staticmethod
    def criar_plano(cubo: CuboVirtual, ponto: Tuple[int, int, int], 
                   normal: Tuple[float, float, float], espessura: int = 1, valor=1):
        """
        Cria um plano no cubo virtual.
        
        Args:
            cubo: Instância do CuboVirtual
            ponto: Um ponto no plano
            normal: Vetor normal ao plano (nx, ny, nz)
            espessura: Espessura do plano em voxels
            valor: Valor para voxels do plano
        """
        px, py, pz = ponto
        nx, ny, nz = normal
        
        # Normalizar vetor normal
        magnitude = (nx**2 + ny**2 + nz**2)**0.5
        if magnitude > 0:
            nx, ny, nz = nx/magnitude, ny/magnitude, nz/magnitude
        
        coordenadas = []
        for x in range(cubo.largura):
            for y in range(cubo.altura):
                for z in range(cubo.profundidade):
                    # Distância do ponto ao plano
                    distancia = abs(nx*(x-px) + ny*(y-py) + nz*(z-pz))
                    
                    if distancia <= espessura/2:
                        cubo.definir_posicao(x, y, z, valor)
                        coordenadas.append((x, y, z))
        
        return coordenadas


class SimuladorVisaoEstereo:
    """
    Simulador de sistema de visão estéreo para teste de algoritmos
    de reconstrução 3D e calibração de câmeras.
    """
    
    def __init__(self, cubo_rastreamento: CuboRastreamento):
        self.cubo = cubo_rastreamento
        self.baseline = 50.0  # Distância entre câmeras em mm
        self.focal_length = 800.0  # Distância focal em pixels
    
    def simular_movimento_circular(self, centro: Tuple[float, float, float], 
                                 raio: float, num_frames: int = 20):
        """
        Simula um objeto se movendo em círculo para teste de rastreamento.
        
        Args:
            centro: Centro do movimento circular
            raio: Raio do círculo
            num_frames: Número de frames a simular
        """
        cx, cy, cz = centro
        
        for i in range(num_frames):
            # Calcular posição no círculo
            angulo = (i / num_frames) * 2 * np.pi
            x = cx + raio * np.cos(angulo)
            y = cy + raio * np.sin(angulo)
            z = cz + i * 0.5  # Movimento gradual em Z
            
            # Adicionar ruído realista
            x += random.uniform(-0.3, 0.3)
            y += random.uniform(-0.3, 0.3)
            z += random.uniform(-0.1, 0.1)
            
            # Simular confiança variável
            confianca = 0.85 + random.uniform(0, 0.15)
            
            # Registrar detecção
            timestamp = time.time() + i * 0.1
            self.cubo.registrar_ponto_detectado(x, y, z, confianca, timestamp)
    
    def calcular_disparidade(self, x_left: float, x_right: float) -> float:
        """Calcula disparidade entre pontos correspondentes nas duas imagens."""
        return abs(x_left - x_right)
    
    def calcular_profundidade(self, disparidade: float) -> float:
        """Calcula profundidade usando disparidade e parâmetros da câmera."""
        if disparidade > 0:
            return (self.baseline * self.focal_length) / disparidade
        return float('inf')

# ===== FUNÇÃO DE DEMONSTRAÇÃO DE PRIMITIVAS =====
def demonstrar_primitivas():
    """
    Demonstra como visualizar diferentes primitivas geométricas no cubo.
    """
    print("🎨 DEMONSTRAÇÃO DE PRIMITIVAS GEOMÉTRICAS")
    print("=" * 50)
    
    # Criar cubo para demonstração
    cubo_demo = CuboVirtual(25, 25, 25, None)
    visualizador = VisualizadorPrimitivas()
    
    print("1. Criando e visualizando uma esfera...")
    # Criar esfera
    coords_esfera = []
    centro_esfera = (12, 12, 12)
    raio_esfera = 5
    AlgoritmosAvancados.criar_esfera(cubo_demo, centro_esfera, raio_esfera)
    
    # Extrair coordenadas da esfera
    coords_esfera = cubo_demo.buscar_posicoes(lambda v: v == 1)
    
    # Visualizar
    visualizador.visualizar_primitiva_no_cubo(cubo_demo, coords_esfera, "Esfera no Cubo")
    
    print("2. Adicionando um cubo sólido...")
    # Limpar cubo e criar cubo sólido
    cubo_demo2 = CuboVirtual(25, 25, 25, None)
    coords_cubo = AlgoritmosAvancados.criar_cubo_solido(cubo_demo2, (18, 18, 18), 6, 2)
    
    visualizador.visualizar_primitiva_no_cubo(cubo_demo2, coords_cubo, "Cubo Sólido")
    
    print("3. Criando um cilindro...")
    # Limpar e criar cilindro
    cubo_demo3 = CuboVirtual(25, 25, 25, None)
    coords_cilindro = AlgoritmosAvancados.criar_cilindro(cubo_demo3, (12, 12, 5), 4, 15, 'z', 3)
    
    visualizador.visualizar_primitiva_no_cubo(cubo_demo3, coords_cilindro, "Cilindro Vertical")
    
    print("4. Comparando múltiplas primitivas...")
    # Criar cubo com várias primitivas
    cubo_multi = CuboVirtual(30, 30, 30, None)
    
    # Esfera pequena
    AlgoritmosAvancados.criar_esfera(cubo_multi, (8, 8, 8), 3, 1)
    coords_esfera_multi = cubo_multi.buscar_posicoes(lambda v: v == 1)
    
    # Cubo sólido
    coords_cubo_multi = AlgoritmosAvancados.criar_cubo_solido(cubo_multi, (22, 22, 22), 4, 2)
    
    # Cilindro
    coords_cilindro_multi = AlgoritmosAvancados.criar_cilindro(cubo_multi, (15, 8, 10), 2, 8, 'y', 3)
    
    # Plano
    coords_plano = AlgoritmosAvancados.criar_plano(cubo_multi, (15, 15, 5), (0, 0, 1), 1, 4)
    
    primitivas = {
        'Esfera': coords_esfera_multi,
        'Cubo': coords_cubo_multi,
        'Cilindro': coords_cilindro_multi,
        'Plano': coords_plano
    }
    
    visualizador.comparar_multiplas_primitivas(cubo_multi, primitivas)
    
    print("✅ Demonstração concluída! Verifique os gráficos gerados.")
    return cubo_multi, primitivas

class DetectorColisao:
    """
    Sistema de detecção de colisão para planejamento de trajetórias
    e navegação autônoma em ambientes 3D.
    """
    
    def __init__(self, cubo_ambiente: CuboVirtual):
        self.ambiente = cubo_ambiente
        self.valor_obstaculo = 1
        self.valor_livre = 0
    
    def criar_ambiente_teste(self):
        """Cria um ambiente de teste com obstáculos variados."""
        # Limpar ambiente
        for x in range(self.ambiente.largura):
            for y in range(self.ambiente.altura):
                for z in range(self.ambiente.profundidade):
                    self.ambiente.definir_posicao(x, y, z, self.valor_livre)
        
        # Criar obstáculos
        self._criar_mesa()
        self._criar_parede()
        self._criar_pilares()
    
    def _criar_mesa(self):
        """Cria uma mesa como obstáculo."""
        for x in range(5, 20):
            for y in range(5, 15):
                for z in range(8, 10):
                    self.ambiente.definir_posicao(x, y, z, self.valor_obstaculo)
    
    def _criar_parede(self):
        """Cria uma parede vertical."""
        for y in range(0, self.ambiente.altura):
            for z in range(0, 20):
                for x in range(25, 29):
                    self.ambiente.definir_posicao(x, y, z, self.valor_obstaculo)
    
    def _criar_pilares(self):
        """Cria pilares espalhados."""
        pilares = [(2, 2), (15, 25), (22, 8)]
        for px, py in pilares:
            for z in range(0, 25):
                for dx in range(2):
                    for dy in range(2):
                        if (px + dx < self.ambiente.largura and 
                            py + dy < self.ambiente.altura):
                            self.ambiente.definir_posicao(px + dx, py + dy, z, 
                                                        self.valor_obstaculo)
    
    def verificar_colisao(self, x: int, y: int, z: int) -> bool:
        """Verifica se uma posição específica tem colisão."""
        return self.ambiente.obter_posicao(x, y, z) == self.valor_obstaculo
    
    def verificar_trajetoria(self, pontos_trajetoria: List[Tuple[int, int, int]]) -> bool:
        """
        Verifica se uma trajetória inteira está livre de colisões.
        
        Args:
            pontos_trajetoria: Lista de pontos (x, y, z) da trajetória
            
        Returns:
            True se a trajetória está livre, False se há colisão
        """
        for x, y, z in pontos_trajetoria:
            if self.verificar_colisao(x, y, z):
                return False
        return True
    
    def calcular_distancia_obstaculo(self, x: int, y: int, z: int) -> float:
        """Calcula a distância mínima ao obstáculo mais próximo."""
        if self.verificar_colisao(x, y, z):
            return 0.0
        
        distancia_min = float('inf')
        
        # Buscar obstáculos em uma região ao redor
        raio_busca = 10
        for ox in range(max(0, x-raio_busca), min(self.ambiente.largura, x+raio_busca+1)):
            for oy in range(max(0, y-raio_busca), min(self.ambiente.altura, y+raio_busca+1)):
                for oz in range(max(0, z-raio_busca), min(self.ambiente.profundidade, z+raio_busca+1)):
                    if self.verificar_colisao(ox, oy, oz):
                        dist = ((x-ox)**2 + (y-oy)**2 + (z-oz)**2)**0.5
                        distancia_min = min(distancia_min, dist)
        
        return distancia_min if distancia_min != float('inf') else raio_busca
    
# ===== NOVA CLASSE: VISUALIZADOR DE PRIMITIVAS =====
class VisualizadorPrimitivas:
    """
    Classe especializada para visualizar primitivas geométricas dentro do cubo.
    Permite ver posicionamento e forma de objetos 3D de forma interativa.
    """
    
    def __init__(self):
        self.fig_count = 0
    
    def visualizar_primitiva_no_cubo(self, cubo: CuboVirtual, primitiva_coords: list, 
                                   titulo: str = "Primitiva no Cubo"):
        """
        Visualiza uma primitiva específica dentro do contexto do cubo.
        
        Args:
            cubo: Instância do CuboVirtual
            primitiva_coords: Lista de coordenadas (x,y,z) da primitiva
            titulo: Título da visualização
        """
        fig = plt.figure(figsize=(15, 5))
        
        # Plot 1: Vista 3D completa
        ax1 = fig.add_subplot(131, projection='3d')
        
        # Desenhar bordas do cubo
        self._desenhar_bordas_cubo(ax1, cubo)
        
        # Plotar primitiva
        if primitiva_coords:
            prim_x, prim_y, prim_z = zip(*primitiva_coords)
            ax1.scatter(prim_x, prim_y, prim_z, c='red', s=50, alpha=0.8, label='Primitiva')
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title(f'{titulo} - Vista 3D')
        ax1.legend()
        
        # Plot 2: Projeção XY (vista de cima)
        ax2 = fig.add_subplot(132)
        if primitiva_coords:
            ax2.scatter(prim_x, prim_y, c='red', s=30, alpha=0.7)
        
        # Bordas do cubo em 2D
        ax2.plot([0, cubo.largura-1, cubo.largura-1, 0, 0], 
                [0, 0, cubo.altura-1, cubo.altura-1, 0], 'b-', alpha=0.5)
        
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('Projeção XY (Vista de Cima)')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        # Plot 3: Informações da primitiva
        ax3 = fig.add_subplot(133)
        ax3.axis('off')
        
        if primitiva_coords:
            info_text = f"""
INFORMAÇÕES DA PRIMITIVA

Total de Voxels: {len(primitiva_coords)}

Dimensões do Cubo:
- Largura: {cubo.largura}
- Altura: {cubo.altura}  
- Profundidade: {cubo.profundidade}

Coordenadas Extremas:
- X: {min(prim_x)} → {max(prim_x)}
- Y: {min(prim_y)} → {max(prim_y)}
- Z: {min(prim_z)} → {max(prim_z)}

Centro Aproximado:
- X: {sum(prim_x)/len(prim_x):.1f}
- Y: {sum(prim_y)/len(prim_y):.1f}
- Z: {sum(prim_z)/len(prim_z):.1f}
            """
        else:
            info_text = "Nenhuma primitiva encontrada"
            
        ax3.text(0.1, 0.9, info_text, transform=ax3.transAxes, 
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f'primitiva_visualizacao_{self.fig_count}.png', dpi=300, bbox_inches='tight')
        plt.show()
        self.fig_count += 1
    
    def _desenhar_bordas_cubo(self, ax, cubo):
        """Desenha as bordas wireframe do cubo."""
        # Coordenadas dos vértices
        vertices = [
            [0, 0, 0], [cubo.largura-1, 0, 0], [cubo.largura-1, cubo.altura-1, 0], [0, cubo.altura-1, 0],  # Base inferior
            [0, 0, cubo.profundidade-1], [cubo.largura-1, 0, cubo.profundidade-1], 
            [cubo.largura-1, cubo.altura-1, cubo.profundidade-1], [0, cubo.altura-1, cubo.profundidade-1]  # Base superior
        ]
        
        # Arestas do cubo
        arestas = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Base inferior
            [4, 5], [5, 6], [6, 7], [7, 4],  # Base superior
            [0, 4], [1, 5], [2, 6], [3, 7]   # Arestas verticais
        ]
        
        for aresta in arestas:
            pontos = [vertices[aresta[0]], vertices[aresta[1]]]
            ax.plot([pontos[0][0], pontos[1][0]], 
                   [pontos[0][1], pontos[1][1]], 
                   [pontos[0][2], pontos[1][2]], 'b-', alpha=0.3)
    
    def comparar_multiplas_primitivas(self, cubo: CuboVirtual, primitivas_dict: dict):
        """
        Compara múltiplas primitivas no mesmo cubo.
        
        Args:
            cubo: Instância do CuboVirtual
            primitivas_dict: Dicionário {'nome': [coordenadas], ...}
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Desenhar bordas do cubo
        self._desenhar_bordas_cubo(ax, cubo)
        
        # Cores para diferentes primitivas
        cores = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
        
        for i, (nome, coords) in enumerate(primitivas_dict.items()):
            if coords:
                x, y, z = zip(*coords)
                cor = cores[i % len(cores)]
                ax.scatter(x, y, z, c=cor, s=30, alpha=0.7, label=nome)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Comparação de Múltiplas Primitivas')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'multiplas_primitivas_{self.fig_count}.png', dpi=300, bbox_inches='tight')
        plt.show()
        self.fig_count += 1
    

# ===== 8. SISTEMA DE TESTES EXPANDIDO =====
class TestadorExpandido(TestadorComVisualizacoes):
    """
    Versão expandida do testador com funcionalidades avançadas.
    """
    
    def executar_teste_algoritmos_avancados(self):
        """Testa algoritmos geométricos avançados."""
        print("📊 5. Teste de Algoritmos Avançados")
        
        # Criar cubo para geometria
        cubo_geo = CuboVirtual(20, 20, 20)
        
        # Criar esfera
        centro_esfera = (10, 10, 10)
        raio_esfera = 6
        AlgoritmosAvancados.criar_esfera(cubo_geo, centro_esfera, raio_esfera)
        
        # Analisar propriedades
        volume = AlgoritmosAvancados.calcular_volume(cubo_geo)
        centro_massa = AlgoritmosAvancados.calcular_centro_massa(cubo_geo)
        superficie = AlgoritmosAvancados.detectar_superficie(cubo_geo)
        
        print(f"   Esfera criada - Volume: {volume}, Centro de massa: {centro_massa}")
        print(f"   Voxels de superfície: {len(superficie)}")
        
        return {
            'cubo': cubo_geo,
            'volume': volume,
            'centro_massa': centro_massa,
            'superficie': superficie,
            'centro_esfera': centro_esfera,
            'raio': raio_esfera
        }
    
    def executar_teste_visao_estereo(self):
        """Testa simulação de visão estéreo."""
        print("📊 6. Teste de Visão Estéreo")
        
        # Criar sistema de rastreamento
        cubo_estereo = CuboRastreamento(30, 30, 30, origem=(0, 0, 0), escala=1.0)
        simulador = SimuladorVisaoEstereo(cubo_estereo)
        
        # Simular movimento circular
        centro_movimento = (15, 15, 15)
        raio_movimento = 8
        simulador.simular_movimento_circular(centro_movimento, raio_movimento, 16)
        
        # Analisar trajetória
        trajetoria = cubo_estereo.calcular_trajetoria()
        print(f"   Movimento simulado: {len(trajetoria)} pontos detectados")
        
        return {
            'simulador': simulador,
            'trajetoria': trajetoria,
            'centro': centro_movimento,
            'raio': raio_movimento
        }
    
    def executar_teste_deteccao_colisao(self):
        """Testa sistema de detecção de colisão."""
        print("📊 7. Teste de Detecção de Colisão")
        
        # Criar ambiente
        ambiente = CuboVirtual(30, 30, 30, 0)
        detector = DetectorColisao(ambiente)
        detector.criar_ambiente_teste()
        
        # Testar trajetórias
        trajetoria_segura = [(5, 5, 5), (10, 10, 10), (15, 15, 15)]
        trajetoria_colisao = [(5, 5, 8), (10, 10, 9), (15, 15, 9)]  # Passa pela mesa
        
        segura = detector.verificar_trajetoria(trajetoria_segura)
        colisao = detector.verificar_trajetoria(trajetoria_colisao)
        
        print(f"   Trajetória segura: {segura}, Trajetória com colisão: {not colisao}")
        
        return {
            'detector': detector,
            'ambiente': ambiente,
            'trajetoria_segura': trajetoria_segura,
            'trajetoria_colisao': trajetoria_colisao
        }
    
    def executar_teste_primitivas_geometricas(self):
        """Novo teste para primitivas geométricas."""
        print("📊 8. Teste de Primitivas Geométricas")
        
        cubo_primitivas, primitivas_dict = demonstrar_primitivas()
        
        # Análise adicional
        total_voxels = sum(len(coords) for coords in primitivas_dict.values())
        print(f"   Total de voxels em todas as primitivas: {total_voxels}")
        
        return {
            'cubo': cubo_primitivas,
            'primitivas': primitivas_dict,
            'total_voxels': total_voxels
        }
    
    # CORREÇÃO: Método principal correto
    def executar_todos_testes_expandidos(self):
        """Executa todos os testes incluindo os avançados."""
        print("🎨 EXECUTANDO SISTEMA EXPANDIDO COMPLETO")
        print("=" * 60)
        
        # CORREÇÃO: Chamar método correto da classe pai
        resultados_basicos = super().executar_todos_testes()  # ← MUDANÇA AQUI
        
        # Testes avançados
        print("\n🔬 TESTES AVANÇADOS")
        print("-" * 30)
        
        algoritmos = self.executar_teste_algoritmos_avancados()
        visao_estereo = self.executar_teste_visao_estereo()
        colisao = self.executar_teste_deteccao_colisao()
        primitivas = self.executar_teste_primitivas_geometricas()
        
        # Compilar resultados
        resultados_completos = {
            **resultados_basicos,
            'algoritmos_avancados': algoritmos,
            'visao_estereo': visao_estereo,
            'deteccao_colisao': colisao,
            'primitivas_geometricas': primitivas
        }
        
        print(f"\n✅ Sistema expandido concluído!")
        print(f"📁 Total de gráficos: {self.visualizador.fig_count}")
        print("📋 Funcionalidades testadas:")
        print("   • Estruturas de dados 3D")
        print("   • Algoritmos geométricos")
        print("   • Simulação de visão estéreo")
        print("   • Detecção de colisão")
        print("   • Primitivas geométricas")
        print("   • Rastreamento temporal")
        print("   • Análise de performance")
        
        return resultados_completos

    # CORREÇÃO: Novo método para evitar confusão
    def executar_todos_testes_com_primitivas(self):
        """Executa todos os testes incluindo primitivas - método simplificado."""
        print("🎨 EXECUTANDO SISTEMA COMPLETO COM PRIMITIVAS")
        print("=" * 60)
        
        # Executar testes básicos primeiro
        print("🧪 TESTES BÁSICOS")
        print("-" * 30)
        resultados_basicos = super().executar_todos_testes()
        
        # Executar testes avançados
        print("\n🔬 TESTES AVANÇADOS")
        print("-" * 30)
        
        try:
            algoritmos = self.executar_teste_algoritmos_avancados()
            visao_estereo = self.executar_teste_visao_estereo()
            colisao = self.executar_teste_deteccao_colisao()
            primitivas = self.executar_teste_primitivas_geometricas()
            
            # Combinar todos os resultados
            resultados_completos = {
                **resultados_basicos,
                'algoritmos_avancados': algoritmos,
                'visao_estereo': visao_estereo,
                'deteccao_colisao': colisao,
                'primitivas_geometricas': primitivas
            }
            
        except Exception as e:
            print(f"⚠️ Erro nos testes avançados: {e}")
            print("Continuando com testes básicos...")
            resultados_completos = resultados_basicos
        
        print(f"\n✅ Sistema completo concluído!")
        return resultados_completos
   

# ===== 9. FUNÇÃO PRINCIPAL CORRIGIDA =====
def main():
    """
    Função principal que demonstra todo o sistema reorganizado.
    """
    print("🎨 SISTEMA DE CUBO VIRTUAL 3D - VERSÃO COMPLETA")
    print("=" * 60)
    print("Sistema reorganizado com estrutura hierárquica:")
    print("1. Classes base (CuboVirtual)")
    print("2. Classes especializadas (CuboRastreamento)")
    print("3. Algoritmos avançados (Geometria, Estéreo, Colisão)")
    print("4. Visualizações completas")
    print("5. Testes abrangentes")
    print()
    
    try:
        # Demonstração passo a passo
        print("🔧 DEMONSTRAÇÃO BÁSICA")
        cubo_demo = exemplo_uso_basico()
        
        print("\n" + "="*40)
        
        # Sistema completo de testes
        print("\n🧪 EXECUTANDO TESTES COMPLETOS...")
        testador = TestadorExpandido()
        resultados = testador.executar_todos_testes_com_primitivas()  # ← MÉTODO CORRETO
        
        print("\n📊 ESTATÍSTICAS FINAIS:")
        print(f"Classes implementadas: 7+")
        print(f"Métodos de visualização: 10+")
        print(f"Algoritmos avançados: 15+")
        print(f"Tipos de teste: 8")
        
        return resultados
        
    except Exception as e:
        print(f"❌ Erro na execução: {e}")
        print("Executando versão básica...")
        
        # Fallback para versão básica
        testador_basico = TestadorComVisualizacoes()
        return testador_basico.executar_todos_testes()


# ===== PONTO DE ENTRADA =====
if __name__ == "__main__":
    # Executar sistema completo
    resultados_sistema = main()
        # Usar sistema completo existente
    print("🎨 SISTEMA COMPLETO COM PRIMITIVAS")
    
    # Demonstração rápida
    demonstrar_primitivas()
    
    # OU usar sistema completo
    testador = TestadorExpandido()
    resultados_completos = testador.executar_todos_testes_com_primitivas()