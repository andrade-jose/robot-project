# Importações de bibliotecas
import os
import caer
import canaro
import numpy as np
import cv2 as cv
import gc
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
import math
from tensorflow.keras.losses import CategoricalCrossentropy

# =============================================
# PARÂMETROS GLOBAIS
# =============================================
IMG_SIZE = (80, 80)          # Tamanho das imagens (altura, largura)
channels = 1                 # 1 canal = imagens em escala de cinza
BATCH_SIZE = 128             # Número de imagens por lote durante o treino
EPOCHS = 50                  # Número de épocas de treinamento
IMG_SHAPE = (80, 80, 1)      # Formato da entrada da rede neural
output_dim = 10              # Número de classes (personagens)

# =============================================
# CARREGAMENTO E PREPARAÇÃO DOS DADOS
# =============================================

# Caminho para o dataset
char_path = r'/kaggle/input/the-simpsons-characters-dataset/simpsons_dataset'

# Cria dicionário com contagem de imagens por personagem
char_dict = {}
for char in os.listdir(char_path):
    char_dict[char] = len(os.listdir(os.path.join(char_path, char)))

# Ordena personagens por quantidade de imagens (decrescente)
char_dict = caer.sort_dict(char_dict, descending=True)

# Seleciona os 10 personagens com mais imagens
characters = []
count = 0
for i in char_dict:
    characters.append(i[0])
    count += 1
    if count >= 10:
        break
print("Personagens selecionados:", characters)

# Pré-processamento das imagens
train = caer.preprocess_from_dir(char_path, characters, channels=channels, IMG_SIZE=IMG_SIZE, isShuffle=True)

# Separa features (imagens) e labels (classes)
featureSet, labels = caer.sep_train(train, IMG_SIZE=IMG_SIZE)
del train  # Libera memória

# Converte labels para one-hot encoding (ex: [0,1,0,0,...])
labels = to_categorical(labels, len(characters))

# Divide em conjuntos de treino e validação (80% treino, 20% validação)
x_train, x_val, y_train, y_val = caer.train_val_split(featureSet, labels, val_ratio=0.2)

# Limpeza de memória
del featureSet
gc.collect()

# Normaliza valores dos pixels para [0,1]
x_train = x_train / 255.0
x_val = x_val / 255.0

# =============================================
# CONSTRUÇÃO DO MODELO
# =============================================

model = Sequential([
    Input(shape=IMG_SHAPE),  # Camada de entrada
    
    # Bloco 1 - Extração de características
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),  # Normaliza os ativados para acelerar treino
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),  # Redução dimensional
    Dropout(0.2),          # Prevenção de overfitting
    
    # Bloco 2
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    
    # Bloco 3
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.4),
    
    # Camadas Densas (classificação)
    Flatten(),  # Achata os features maps para vetor
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(output_dim, activation='softmax')  # Saída com probabilidades
])

# Otimizador SGD com momentum
optimizer = SGD(
    learning_rate=0.01,  # Taxa de aprendizado inicial
    momentum=0.9,       # Suaviza atualizações dos pesos
    nesterov=True       # Variante mais eficiente do momentum
)

# Compila o modelo
model.compile(
    optimizer=optimizer,
    loss=CategoricalCrossentropy(),  # Função de perda para classificação
    metrics=['accuracy']             # Acompanha a acurácia
)

# =============================================
# AUMENTO DE DADOS (DATA AUGMENTATION)
# =============================================
datagen = ImageDataGenerator(
    rotation_range=30,       # Rotação aleatória até 30 graus
    width_shift_range=0.2,   # Deslocamento horizontal
    height_shift_range=0.2,  # Deslocamento vertical
    shear_range=0.2,         # Deformação angular
    zoom_range=0.2,          # Zoom aleatório
    horizontal_flip=True,    # Espelhamento horizontal
    fill_mode='nearest'      # Preenche pixels faltantes
)

# =============================================
# CALLBACKS (CONTROLE DE TREINAMENTO)
# =============================================

# Agenda redução da taxa de aprendizado
def lr_schedule(epoch):
    initial_lr = 0.01
    drop = 0.5       # Fator de redução
    epochs_drop = 10  # Reduz a cada 10 épocas
    return initial_lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))

callbacks_list = [
    LearningRateScheduler(lr_schedule),  # Ajusta taxa de aprendizado
    EarlyStopping(                      # Para treino se não melhorar
        monitor='val_loss', 
        patience=15, 
        restore_best_weights=True
    ),
    ModelCheckpoint(                    # Salva melhor modelo
        'best_model.h5', 
        monitor='val_accuracy', 
        save_best_only=True
    )
]

# =============================================
# TREINAMENTO DO MODELO
# =============================================
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True),
    steps_per_epoch=len(x_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_val, y_val),
    callbacks=callbacks_list,
    verbose=1
)

# =============================================
# FUNÇÃO PARA PREPARAR NOVAS IMAGENS
# =============================================
def prepare(img):
    # Converte para escala de cinza
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  
    # Redimensiona
    img = cv.resize(img, IMG_SIZE)  
    # Reformata para (80,80,1)
    img = caer.reshape(img, IMG_SIZE, 1)  
    # Normaliza
    img = img / 255.0  
    # Adiciona dimensão do batch (1,80,80,1)
    return np.expand_dims(img, axis=0)  

# =============================================
# TESTE COM UMA IMAGEM EXEMPLO
# =============================================
test_path = r'../input/the-simpsons-characters-dataset/kaggle_simpson_testset/kaggle_simpson_testset/charles_montgomery_burns_0.jpg'
img = cv.imread(test_path)

# Mostra a imagem
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Faz a predição
prepared_img = prepare(img)
predictions = model.predict(prepared_img)
print("Personagem previsto:", characters[np.argmax(predictions[0])])