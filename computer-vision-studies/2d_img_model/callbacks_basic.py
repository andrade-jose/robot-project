import os
import math
from tensorflow.keras.callbacks import (
    LearningRateScheduler, 
    EarlyStopping, 
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard
)
from config_basic import config
from datetime import datetime

def lr_schedule(epoch):
    """Agendamento de taxa de aprendizado com decaimento exponencial"""
    initial_lr = config.INITIAL_LR
    drop = 0.5
    epochs_drop = 10
    lr = initial_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lr

def get_training_callbacks(model_name='shape_classifier'):
    """
    Retorna lista completa de callbacks para treinamento
    
    Args:
        model_name: Nome do modelo para salvar os checkpoints
        
    Returns:
        Lista de callbacks configurados
    """
    # Garante que os diretórios existam
    os.makedirs(config.TRAINED_MODELS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    
    callbacks = [
        # Agendamento de learning rate
        LearningRateScheduler(lr_schedule, verbose=1),
        
        # Early stopping para evitar overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Checkpoint do modelo
        ModelCheckpoint(
            filepath=os.path.join(config.TRAINED_MODELS_DIR, f'{model_name}.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        ),
        
        # Redução dinâmica de LR
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        
        # Logs para TensorBoard
        TensorBoard(
            log_dir=config.LOGS_DIR,
            histogram_freq=1,
            update_freq='epoch'
        )
    ]
    
    return callbacks