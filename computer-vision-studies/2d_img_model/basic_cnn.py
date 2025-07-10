from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, ReLU, GlobalAveragePooling2D
)
from tensorflow.keras.regularizers import l2
from config_basic import config

def build_basic_cnn(input_shape,num_classes, dropout_rate=0.5):
    model = Sequential([
        Input(shape=input_shape),
        
        Conv2D(32, 3, padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        ReLU(),
        Conv2D(32, 3, padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        ReLU(),
        MaxPooling2D(2),
        Dropout(dropout_rate * 0.5),
        
        Conv2D(64, 3, padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        ReLU(),
        Conv2D(64, 3, padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        ReLU(),
        MaxPooling2D(2),
        Dropout(dropout_rate),

        Conv2D(128, 3, padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        ReLU(),
        Conv2D(128, 3, padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        ReLU(),
        MaxPooling2D(2),
        Dropout(dropout_rate),

        GlobalAveragePooling2D(),
        Dense(512, activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax', dtype='float32')
    ])
    return model
