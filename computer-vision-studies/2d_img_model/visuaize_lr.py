import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import img_to_array, load_img

# ======================
# 1. Criar modelo
# ======================
def build_model(use_regularizer=False, use_dropout=False):
    input_layer = Input(shape=(64, 64, 1))
    
    x = Conv2D(32, (3, 3), activation='relu', 
               kernel_regularizer=l2(0.01) if use_regularizer else None)(input_layer)
    if use_dropout:
        x = Dropout(0.5)(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu', 
               kernel_regularizer=l2(0.01) if use_regularizer else None)(x)
    model = Model(inputs=input_layer, outputs=x)
    return model

# ======================
# 2. Carregar e preprocessar imagem
# ======================
img_path = r'C:\Venv\Robot_project\datasets\dataset opencv\photos\cat.jpg'
img = load_img(img_path, color_mode='grayscale', target_size=(64, 64))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalização opcional

# ======================
# 3. Comparar modelos
# ======================
models = {
    "Sem Regularização": build_model(use_regularizer=False),
    "Com L2 Regularization": build_model(use_regularizer=True),
}

# ======================
# 4. Visualizar Feature Maps
# ======================
for name, model in models.items():
    print(f"\n🔍 {name}:")
    
    # Criar modelo para visualização da primeira camada convolucional
    activation_model = Model(inputs=model.input, outputs=model.layers[1].output)
    
    # Obter os mapas de ativação
    feature_maps = activation_model.predict(img_array)
    
    plt.figure(figsize=(12, 8))
    plt.suptitle(f"Feature Maps - {name}", fontsize=16)
    
    for i in range(min(16, feature_maps.shape[-1])):
        plt.subplot(4, 4, i+1)
        plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
