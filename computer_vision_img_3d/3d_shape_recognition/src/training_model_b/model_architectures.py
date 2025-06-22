from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, BatchNormalization
)
from src.utils.config_b import config

def build_basic_cnn(input_shape=config.IMG_SHAPE, num_classes=6):
    """Constrói a arquitetura básica do modelo CNN"""
    model = Sequential([
        Input(shape=input_shape),
        
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
        Dense(config.NUM_CLASSES, activation='softmax')
    ])
    
    return model

def build_advanced_cnn(input_shape=config.IMG_SHAPE, num_classes=6):
    """Arquitetura CNN mais avançada"""
    model = Sequential([
        # ... implementação da arquitetura avançada ...
    ])
    return model