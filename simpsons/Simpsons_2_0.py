import os
import math
import gc

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import caer
import canaro

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten,
                                     Dense, Dropout, BatchNormalization)
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint


# --- Parâmetros ---
IMG_SIZE = (80, 80)
CHANNELS = 1
IMG_SHAPE = (80, 80, 1)
OUTPUT_DIM = 10
BATCH_SIZE = 128
EPOCHS = 50

# --- Preparação dos dados ---
char_path = r'/kaggle/input/the-simpsons-characters-dataset/simpsons_dataset'

# Contar imagens por personagem
char_dict = {}
for char in os.listdir(char_path):
    char_dict[char] = len(os.listdir(os.path.join(char_path, char)))

# Ordenar personagens pelo número de imagens (decrescente)
char_dict = caer.sort_dict(char_dict, descending=True)

# Selecionar os 10 personagens com mais imagens
characters = [c[0] for c in char_dict[:OUTPUT_DIM]]

# Preprocessar imagens
train = caer.preprocess_from_dir(char_path, characters, channels=CHANNELS, IMG_SIZE=IMG_SIZE, isShuffle=True)
featureSet, labels = caer.sep_train(train, IMG_SIZE=IMG_SIZE)
del train

labels = to_categorical(labels, num_classes=OUTPUT_DIM)

# Separar treino e validação
x_train, x_val, y_train, y_val = caer.train_val_split(featureSet, labels, val_ratio=0.2)
del featureSet
gc.collect()

# --- Definição do modelo ---
model = Sequential([
    Input(shape=IMG_SHAPE),

    # Bloco 1
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    # Bloco 2
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    # Bloco 3
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.4),

    # Camadas densas
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(OUTPUT_DIM, activation='softmax')
])

model.summary()

# --- Otimizador SGD com agendamento de learning rate ---
optimizer = SGD(
    learning_rate=0.01,
    momentum=0.9,
    nesterov=True,
    weight_decay=1e-4  # regularização L2
)

model.compile(
    optimizer=optimizer,
    loss=CategoricalCrossentropy(),
    metrics=['accuracy']
)

# --- Data Augmentation ---
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    rescale=1./255
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow(
    x_train,
    y_train,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_gen = val_datagen.flow(
    x_val,
    y_val,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# --- Learning rate scheduler ---
def lr_schedule(epoch):
    initial_lr = 0.01
    drop = 0.5
    epochs_drop = 10
    lr = initial_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lr

callbacks_list = [
    LearningRateScheduler(lr_schedule),
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)
]

# --- Treinamento ---
history = model.fit(
    train_gen,
    steps_per_epoch=max(1, len(x_train) // BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=val_gen,
    validation_steps=max(1, len(y_val) // BATCH_SIZE),
    callbacks=callbacks_list,
    verbose=1
)

# --- Função para preparar imagem para predição ---
def prepare(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, IMG_SIZE)
    img = caer.reshape(img, IMG_SIZE, CHANNELS)
    img = img.astype('float32') / 255.0  # Normalizar igual ao treinamento
    img = np.expand_dims(img, axis=0)    # Adicionar batch dimension
    return img

# --- Teste ---
test_path = r'../input/the-simpsons-characters-dataset/kaggle_simpson_testset/kaggle_simpson_testset/charles_montgomery_burns_0.jpg'
img = cv.imread(test_path)

plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

predictions = model.predict(prepare(img))
print(f'Predicted character: {characters[np.argmax(predictions[0])]}')
