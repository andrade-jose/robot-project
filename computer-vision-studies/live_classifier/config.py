def get_config(self):
    config = super().get_config()

    return config

# --- Parâmetros ---
IMG_SIZE: tuple = (125, 125)          # Tamanho da imagem (altura, largura)
CHANNELS: int = 3                     # 1 para grayscale, 3 para RGB

# --- Lista das classes ---
characters = ['cubo', 'esfera', 'cone']

model_ok = r'C:\Venv\Rep_git\robot-project\computer-vision-studies\models\trained\multiview_cnn_best.h5'