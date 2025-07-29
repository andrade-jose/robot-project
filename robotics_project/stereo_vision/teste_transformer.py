import numpy as np
import sys
from pathlib import Path

# Add project root to Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from stereo_vision.coordinate_transformer import CoordinateTransformer

def teste_simples_transformacao():
    print("🚀 Iniciando teste de transformação de ponto...")

    # Matriz de transformação fictícia (translação + identidade)
    fake_T = np.array([
        [1, 0, 0, 0.1],
        [0, 1, 0, 0.2],
        [0, 0, 1, 0.3],
        [0, 0, 0, 1]
    ])

    # Instancia o transformador e injeta a matriz
    transformer = CoordinateTransformer()
    transformer.T = fake_T

    # Lista de pontos de entrada simulados
    pontos_camera = [
        [0.0, 0.0, 0.0],
        [0.1, 0.1, 0.1],
        [-0.2, 0.3, 0.4]
    ]

    for i, ponto in enumerate(pontos_camera):
        try:
            ponto_robot = transformer.transform_point(ponto)
            print(f"✅ Ponto {i} transformado: {ponto} -> {ponto_robot}")
        except Exception as e:
            print(f"❌ Erro ao transformar ponto {i}: {e}")

if __name__ == "__main__":
    teste_simples_transformacao()
