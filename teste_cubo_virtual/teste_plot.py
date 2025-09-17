#!/usr/bin/env python3
"""
Visualização 3D das Coordenadas no Cubo
=======================================
Plota as coordenadas dos marcadores detectados em um cubo 3D
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plotar_coordenadas_cubo():
    """Plota as coordenadas dos marcadores em um cubo 3D."""
    
    # Dados das coordenadas detectadas
    grupo_1 = {
        4: (172.8, 8.5, -7.1),
        6: (173.3, 74.8, -4.9),
        2: (175.2, -56.9, 1.8)
    }
    
    grupo_2 = {
        5: (48.5, 2.5, -6.9),
        7: (52.2, 63.1, -8.2),
        3: (48.9, -65.7, -2.9)
    }
    
    # Dimensões do cubo
    largura = 240  # mm
    altura = 240   # mm
    profundidade = 80  # mm
    
    # Criar figura 3D
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Desenhar wireframe do cubo
    desenhar_cubo_wireframe(ax, largura, altura, profundidade)
    
    # Plotar pontos do Grupo 1 (Vermelho)
    for marker_id, (x, y, z) in grupo_1.items():
        ax.scatter(x, y, z, c='red', s=200, marker='o', alpha=0.8, edgecolors='darkred', linewidth=2)
        ax.text(x+5, y+5, z+5, f'G1({marker_id})', fontsize=10, color='darkred', weight='bold')
    
    # Plotar pontos do Grupo 2 (Azul)
    for marker_id, (x, y, z) in grupo_2.items():
        ax.scatter(x, y, z, c='blue', s=200, marker='s', alpha=0.8, edgecolors='darkblue', linewidth=2)
        ax.text(x+5, y+5, z+5, f'G2({marker_id})', fontsize=10, color='darkblue', weight='bold')
    
    # Configurar eixos
    ax.set_xlabel('X (mm)', fontsize=12, weight='bold')
    ax.set_ylabel('Y (mm)', fontsize=12, weight='bold')
    ax.set_zlabel('Z (mm)', fontsize=12, weight='bold')
    
    # Definir limites dos eixos
    ax.set_xlim(0, largura)
    ax.set_ylim(-altura/2, altura/2)  # Centrado em 0
    ax.set_zlim(-40, 40)  # Centrado em 0
    
    # Título
    ax.set_title('Coordenadas dos Marcadores no Cubo 240x240x80mm', fontsize=14, weight='bold', pad=20)
    
    # Legenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.8, label='Grupo 1 (IDs: 2, 4, 6)'),
        Patch(facecolor='blue', alpha=0.8, label='Grupo 2 (IDs: 3, 5, 7)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Mostrar plot
    plt.show()
    
    # Salvar imagem
    plt.savefig('coordenadas_cubo_3d.png', dpi=300, bbox_inches='tight')
    print("Plot salvo como: coordenadas_cubo_3d.png")

def desenhar_cubo_wireframe(ax, largura, altura, profundidade):
    """Desenha o wireframe do cubo."""
    
    # Vértices do cubo (centrado em Y e Z)
    vertices = np.array([
        [0, -altura/2, -profundidade/2],     [largura, -altura/2, -profundidade/2],
        [largura, altura/2, -profundidade/2], [0, altura/2, -profundidade/2],
        [0, -altura/2, profundidade/2],      [largura, -altura/2, profundidade/2],
        [largura, altura/2, profundidade/2],  [0, altura/2, profundidade/2]
    ])
    
    # Definir arestas do cubo
    arestas = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Base inferior
        [4, 5], [5, 6], [6, 7], [7, 4],  # Base superior
        [0, 4], [1, 5], [2, 6], [3, 7]   # Arestas verticais
    ]
    
    # Desenhar arestas
    for aresta in arestas:
        pontos = vertices[aresta]
        ax.plot3D(*pontos.T, 'k-', alpha=0.4, linewidth=1)
    
    # Destacar base e topo
    base_inferior = vertices[[0, 1, 2, 3, 0]]
    base_superior = vertices[[4, 5, 6, 7, 4]]
    
    ax.plot3D(*base_inferior.T, 'g-', alpha=0.6, linewidth=2, label='Base')
    ax.plot3D(*base_superior.T, 'b-', alpha=0.6, linewidth=2, label='Topo')

def imprimir_analise_coordenadas():
    """Imprime análise das coordenadas detectadas."""
    
    grupo_1 = {
        4: (172.8, 8.5, -7.1),
        6: (173.3, 74.8, -4.9),
        2: (175.2, -56.9, 1.8)
    }
    
    grupo_2 = {
        5: (48.5, 2.5, -6.9),
        7: (52.2, 63.1, -8.2),
        3: (48.9, -65.7, -2.9)
    }
    
    print("=== ANÁLISE DAS COORDENADAS DETECTADAS ===")
    print(f"Cubo de trabalho: 240x240x80mm")
    print(f"Sistema de coordenadas: X=0 a 240mm, Y=-120 a +120mm, Z=-40 a +40mm")
    
    print("\nGRUPO 1 (Vermelho):")
    for marker_id, (x, y, z) in grupo_1.items():
        print(f"  ID {marker_id}: X={x:6.1f}mm, Y={y:6.1f}mm, Z={z:5.1f}mm")
    
    print("\nGRUPO 2 (Azul):")
    for marker_id, (x, y, z) in grupo_2.items():
        print(f"  ID {marker_id}: X={x:6.1f}mm, Y={y:6.1f}mm, Z={z:5.1f}mm")
    
    # Análise de distribuição
    print("\n=== ANÁLISE DE DISTRIBUIÇÃO ===")
    
    # Todos os pontos
    todos_pontos = list(grupo_1.values()) + list(grupo_2.values())
    xs = [p[0] for p in todos_pontos]
    ys = [p[1] for p in todos_pontos]
    zs = [p[2] for p in todos_pontos]
    
    print(f"Faixa X: {min(xs):.1f}mm a {max(xs):.1f}mm (amplitude: {max(xs)-min(xs):.1f}mm)")
    print(f"Faixa Y: {min(ys):.1f}mm a {max(ys):.1f}mm (amplitude: {max(ys)-min(ys):.1f}mm)")
    print(f"Faixa Z: {min(zs):.1f}mm a {max(zs):.1f}mm (amplitude: {max(zs)-min(zs):.1f}mm)")
    
    # Observações
    print("\n=== OBSERVAÇÕES ===")
    print("• Grupo 1 está concentrado na região X≈175mm (lado direito)")
    print("• Grupo 2 está concentrado na região X≈50mm (lado esquerdo)")
    print("• Ambos os grupos estão distribuídos ao longo do eixo Y")
    print("• Coordenadas Z são pequenas (próximas ao plano de referência)")
    
    if max(xs) > 240:
        print("⚠️  ATENÇÃO: Alguns pontos estão fora do cubo em X!")
    if max(abs(y) for y in ys) > 120:
        print("⚠️  ATENÇÃO: Alguns pontos estão fora do cubo em Y!")
    if max(abs(z) for z in zs) > 40:
        print("⚠️  ATENÇÃO: Alguns pontos estão fora do cubo em Z!")

def main():
    """Função principal."""
    print("Visualização 3D das Coordenadas no Cubo")
    print("="*50)
    
    # Imprimir análise
    imprimir_analise_coordenadas()
    
    print("\nGerando visualização 3D...")
    
    # Plotar coordenadas
    plotar_coordenadas_cubo()
    
    print("Visualização concluída!")

if __name__ == "__main__":
    main()