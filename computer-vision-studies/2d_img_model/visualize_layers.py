import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Caminhos do projeto
from pathlib import Path
current_dir = Path(__file__).parent
project_root = current_dir
sys.path.append(str(project_root))

# Importações locais
from basic_cnn import build_basic_cnn
from config_basic import config

# =======================
# 1. Carregar uma imagem
# =======================
img_path = r'C:\Venv\Robot_project\datasets\dataset opencv\photos\cat.jpg'  # Altere para o caminho da sua imagem
img = load_img(img_path, target_size=config.IMG_SIZE, color_mode='grayscale')
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # shape = (1, H, W, 1)

# =======================
# 2. Construir o modelo
# =======================
model = build_basic_cnn(input_shape=config.input_shape, num_classes=config.num_classes)

# ================================
# 3. Criar modelo com camadas parciais
# ================================
# Get layer outputs for visualization
layer_outputs = [layer.output for layer in model.layers 
                if 'activation' not in layer.name and 'dropout' not in layer.name]

# Create the activation model
activation_model = Model(inputs=model.inputs, outputs=layer_outputs)  # Note: using model.inputs instead of model.input

# ================================
# 4. Obter ativações
# ================================
activations = activation_model.predict(img_array)

# ================================
# 5. Visualizar ativações
# ================================
for layer_name, activation in zip([layer.name for layer in activation_model.layers], activations):
    if len(activation.shape) == 4:
        n_features = activation.shape[-1]
        size = activation.shape[1]

        n_cols = min(n_features, 8)
        n_rows = int(np.ceil(n_features / n_cols))

        display_grid = np.zeros((size * n_rows, size * n_cols))

        for col in range(n_cols):
            for row in range(n_rows):
                idx = col + n_cols * row
                if idx < n_features:
                    feature_map = activation[0, :, :, idx]
                    feature_map -= feature_map.mean()
                    feature_map /= (feature_map.std() + 1e-5)
                    feature_map *= 64
                    feature_map += 128
                    feature_map = np.clip(feature_map, 0, 255).astype('uint8')
                    display_grid[row * size : (row + 1) * size,
                               col * size : (col + 1) * size] = feature_map

        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(f'Ativações da camada: {layer_name}')
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.axis('off')
        plt.tight_layout()
        plt.show()