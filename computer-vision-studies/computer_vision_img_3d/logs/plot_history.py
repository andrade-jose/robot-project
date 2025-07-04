import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required=True, help='Caminho para o arquivo .npy com histórico')
args = parser.parse_args()

history = np.load(args.file, allow_pickle=True).item()

plt.figure(figsize=(10, 5))
plt.plot(history['accuracy'], label='Treinamento')
plt.plot(history['val_accuracy'], label='Validação')
plt.title('Acurácia ao longo das épocas')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
