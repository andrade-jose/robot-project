import os
import math
import shutil
from datetime import datetime
from tensorflow.keras.callbacks import (
    LearningRateScheduler, 
    EarlyStopping, 
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
    CSVLogger
)
from config.config_advanced import config

def lr_schedule(epoch):
    """Agendamento de taxa de aprendizado com decaimento exponencial adaptado para multiview"""
    initial_lr = config.INITIAL_LR
    drop = 0.5
    epochs_drop = 15  # Aumentado para treinamento mais longo
    min_lr = 1e-6
    
    lr = initial_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return max(lr, min_lr)

def get_training_callbacks(model_name='multiview_cnn', **kwargs):
    """
    Retorna lista completa de callbacks para treinamento da Multiview CNN
    
    Args:
        model_name: Nome do modelo para salvar os checkpoints
        kwargs: Argumentos adicionais como 'include_aux' para nomes de arquivos
        
    Returns:
        Lista de callbacks configurados
    """
    # Diretórios com sufixo para aux features se necessário
    suffix = "_with_aux" if kwargs.get('include_aux', False) else ""
    model_name = f"{model_name}{suffix}"
    
    # Garante que os diretórios existam
    os.makedirs(config.TRAINED_MODELS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    
    # Nome único para o experimento
    experiment_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    callbacks = [
        # Agendamento de learning rate adaptado para treinamento mais longo
        LearningRateScheduler(lr_schedule, verbose=1),
        
        # Early stopping com paciência aumentada para redes complexas
        EarlyStopping(
            monitor='val_loss',
            patience=20,  # Aumentado para multiview
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),
        
        # Checkpoint do modelo (salva estrutura completa)
        ModelCheckpoint(
            filepath=os.path.join(config.TRAINED_MODELS_DIR, f'{model_name}_best.h5'),
            monitor='val_pose_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        
        # Redução dinâmica de LR com parâmetros ajustados
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,  # Redução mais agressiva
            patience=8,
            min_lr=1e-6,
            verbose=1,
            mode='min'
        ),
        
        # Logs para TensorBoard com nome de experimento único
        TensorBoard(
            log_dir=os.path.join(config.LOGS_DIR, experiment_name),
            histogram_freq=1,
            update_freq='epoch',
            profile_batch=0,
            write_graph=True,
            write_images=False
        ),
        
        # Logger para CSV (útil para análise posterior)
        CSVLogger(
            filename=os.path.join(config.LOGS_DIR, f'{experiment_name}.csv'),
            separator=',',
            append=False
        )
    ]
    
    return callbacks