import os
import sys
import tensorflow as tf
import numpy as np
import math

from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime
from pathlib import Path
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import CategoricalCrossentropy

current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from data.loading_dir import prepare_data
from architectures.basic_cnn import build_basic_cnn
from training.callbacks_basic import get_training_callbacks
from data.preprocessing_dir import create_data_generators
from config.config_basic import config

try:
    # --- Preparar dados ---
    x_train, x_val, y_train, y_val = prepare_data()

    # --- Modelo ---
    model = build_basic_cnn(input_shape=config.input_shape, num_classes=config.num_classes)

    # --- Compilação ---
    optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True, weight_decay=1e-4)
    model.compile(optimizer=optimizer, loss=CategoricalCrossentropy(), metrics=['accuracy'])

    # --- Aumento de dados ---
    train_gen, val_gen = create_data_generators(x_train, y_train, x_val, y_val)

    # --- Callbacks ---
    callbacks = get_training_callbacks(model_name="Basic_cnn_model")

    # --- Treinamento ---
    history = model.fit(
        train_gen,
        epochs=config.EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    
except Exception as e:
    print("Erro durante a execução:")
    print(e)
    input("Pressione Enter para sair...")