import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import caer

# Parâmetros
IMG_SIZE = (80, 80)
CHANNELS = 1
OUTPUT_DIM = 6

# Lista de classes na mesma ordem do treinamento
characters = ['cubo', 'esfera', 'cone', 'cilindro', 'paralelepipedo',
              'piramide']

# Carregar modelo treinado
model = load_model(r'C:\Venv\OpenCv\computer-vision-studies\computer_vision_img_3d\3d_shape_recognition\logs\basic_cnn_train\basic_cnn_20250623_121317.h5')

# Função para preparar a imagem
def prepare(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, IMG_SIZE)
    img = caer.reshape(img, IMG_SIZE, CHANNELS)
    img = img.astype('float32') / 255.0
    img = img.reshape(1, IMG_SIZE[0], IMG_SIZE[1], CHANNELS)  # corrigido
    return img


# Caminho da imagem a ser testada
test_path = r'C:\Venv\OpenCv\computer-vision-studies\datasets\shapes_3d\cone\cone_000.png'
img = cv.imread(test_path)

# Fazer predição
predictions = model.predict(prepare(img))
classe_predita = characters[np.argmax(predictions[0])]

# Mostrar imagem com texto
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title(f'Reconhecido: {classe_predita}', fontsize=16, color='green')
plt.axis('off')
plt.show()

print(f'✅ Classe predita: {classe_predita}')
