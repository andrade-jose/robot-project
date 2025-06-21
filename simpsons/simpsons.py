import os
import caer
import canaro
import numpy as np
import cv2 as cv
import gc
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from canaro.generators import imageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
import math
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import CategoricalCrossentropy

# Parâmetros
IMG_SIZE = (80,80)
channels = 1
# Configuração dos hiperparâmetros
BATCH_SIZE = 128  # Tamanho do lote (pode ajustar conforme sua GPU)
EPOCHS = 50      # Número de épocas de treinamento
# Defina as dimensões corretas
IMG_SHAPE = (80, 80, 1)  # 80x80 pixels, 1 canal (grayscale)
output_dim = 10  # Número de classes


char_path = r'/kaggle/input/the-simpsons-characters-dataset/simpsons_dataset'

char_dict = {}
for char in os.listdir(char_path):
    char_dict[char] = len(os.listdir(os.path.join(char_path, char)))

#Classificar em ordem decrescente
char_dict = caer.sort_dict(char_dict, descending=True)
char_dict

# Pega os 10 primeros personagens com mais imagens
characters = []
count = 0
for i in char_dict:
    characters.append(i[0])
    count += 1
    if count >= 10:
        break
characters

# Criar treinamento de dados (Processamento)
train = caer.preprocess_from_dir(char_path, characters, channels=channels, IMG_SIZE=IMG_SIZE, isShuffle=True)

featureSet, labels = caer.sep_train(train, IMG_SIZE=IMG_SIZE)
del train

labels = to_categorical(labels, len(characters))

# Separar treino e validação
x_train, x_val, y_train, y_val = caer.train_val_split(featureSet, labels, val_ratio=0.2)

#apagar execesso de memoria

del featureSet
gc.collect()


# Criação do otimizador Adam com taxa de aprendizado personalizada
optimizer = Adam(
    learning_rate=0.001,  # Taxa de aprendizado inicial
    beta_1=0.9,          # Parâmetro para cálculo da média do gradiente
    beta_2=0.999,        # Parâmetro para cálculo da média do quadrado do gradiente
    epsilon=1e-07        # Termo para evitar divisão por zero
)

# Compilação do modelo
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)  # Melhor que SGD



# Use o ImageDataGenerator padrão do Keras para ter todos os parâmetros de aumento de dados
datagen = ImageDataGenerator(
    rotation_range=30,       # Rotação aleatória até 30 graus
    width_shift_range=0.2,   # Deslocamento horizontal (20% da largura)
    height_shift_range=0.2,  # Deslocamento vertical (20% da altura)
    shear_range=0.2,         # Distorção de cisalhamento
    zoom_range=0.2,          # Zoom aleatório
    horizontal_flip=True,    # Inversão horizontal
    fill_mode='nearest',     # Preenchimento de pixels
    rescale=1./255           # Normalização [0-1]
)

# Gerador de treino
train_gen = datagen.flow(
    x_train,
    y_train,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Gerador de validação (sem aumento de dados)
val_datagen = ImageDataGenerator(rescale=1./255)
val_gen = val_datagen.flow(x_val, y_val, batch_size=BATCH_SIZE)



model = Sequential([
    Input(shape=IMG_SHAPE),  # Camada de entrada com shape correto
    
    # Bloco 1
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),  # 80x80 -> 40x40
    Dropout(0.2),
    
    # Bloco 2
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),  # 40x40 -> 20x20
    Dropout(0.3),
    
    # Bloco 3
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),  # 20x20 -> 10x10
    Dropout(0.4),
    
    # Camadas Densas
    Flatten(),  # 10x10x128 = 12800 elementos
    Dense(512, activation='relu'),  # Reduzido para 512 neurônios
    BatchNormalization(),
    Dropout(0.5),
    Dense(output_dim, activation='softmax')
])

# Verificar modelo
model.summary()


# Configuração do otimizador SGD melhorado
optimizer = SGD(
    learning_rate=0.01,  # Taxa maior para SGD puro
    momentum=0.9,
    nesterov=True,
    weight_decay=1e-4  # Regularização L2 incorporada
)

# Compilação correta para classificação multiclasse
model.compile(
    optimizer=optimizer,
    loss=CategoricalCrossentropy(),  # Para labels inteiros (0, 1, 2,...)
    # Ou loss='categorical_crossentropy' se usar one-hot encoding
    metrics=['accuracy']
)

# 1. Definição do Agendamento de Learning Rate Personalizado
def lr_schedule(epoch):
    initial_lr = 0.01
    drop = 0.5
    epochs_drop = 10
    lr = initial_lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lr

# 2. Callbacks Adicionais para Melhor Controle
callbacks_list = [
    LearningRateScheduler(lr_schedule),
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)
]

# 3. Treinamento
history = model.fit(
    train_gen,
    steps_per_epoch=max(1, len(x_train)//BATCH_SIZE),  # Garante no mínimo 1 passo
    epochs=EPOCHS,
    validation_data=(x_val, y_val),
    validation_steps=max(1, len(y_val)//BATCH_SIZE),  # Garante no mínimo 1 passo
    callbacks=callbacks_list,
    verbose=1
)

# Função para prepara imagem
def prepare(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, IMG_SIZE[:2])
    img = caer.reshape(img, IMG_SIZE[:2], 1)
    return img

# Teste 
test_path = r'../input/the-simpsons-characters-dataset/kaggle_simpson_testset/kaggle_simpson_testset/charles_montgomery_burns_0.jpg'
img = cv.imread(test_path)

plt.imshow(img)
plt.axis('off')
plt.show()


predictions = model.predict(prepare(img))
print(characters[np.argmax(predictions[0])])