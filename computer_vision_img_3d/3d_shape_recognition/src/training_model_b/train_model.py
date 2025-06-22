import os
import tensorflow as tf
import sys
from pathlib import Path
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import CategoricalCrossentropy

# Configuração de paths - CORREÇÃO AQUI
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent  # Sobe para 3d_shape_recognition
sys.path.append(str(project_root))

# Imports absolutos
from src.training_model_b.model_architectures import build_basic_cnn
from src.training_model_b.callbacks import get_training_callbacks
from src.data_processing_b.data_loader import ShapeDataLoader
from src.data_processing_b.data_augmentation import create_data_generators
from src.utils.config_b import config

def main():
    # Garante que os diretórios existam
    print("\n" + "="*50)
    print("Configuração de caminhos:")
    print(f"Projeto: {config.BASE_DIR}")
    print(f"Dados externos: {config.DATA_ROOT}")
    print(f"CSV path: {config.CSV_FILE}")
    print(f"O CSV existe? {os.path.exists(config.CSV_FILE)}")
    print("="*50 + "\n")
    
    os.makedirs(config.TRAINED_MODELS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    
    try:
        # 1. Carregar e preparar dados - CORREÇÃO AQUI
        data_loader = ShapeDataLoader()
        images, labels = data_loader.load_data_from_csv()
        (x_train, y_train), (x_val, y_val), _, num_classes = data_loader.prepare_datasets(images, labels)
        
        # 2. Criar geradores de dados
        train_gen, val_gen = create_data_generators(x_train, y_train, x_val, y_val)
        
        # 3. Construir modelo
        model = build_basic_cnn(num_classes=num_classes)
        
        # 4. Compilar modelo
        optimizer = SGD(
            learning_rate=config.INITIAL_LR,
            momentum=0.9,
            nesterov=True,
            weight_decay=config.WEIGHT_DECAY
        )
        
        model.compile(
            optimizer=optimizer,
            loss=CategoricalCrossentropy(),
            metrics=['accuracy']
        )
        
        # 5. Treinar modelo
        history = model.fit(
            train_gen,
            steps_per_epoch=max(1, len(x_train) // config.BATCH_SIZE),
            epochs=config.EPOCHS,
            validation_data=val_gen,
            validation_steps=max(1, len(y_val) // config.BATCH_SIZE),
            callbacks=get_training_callbacks(),
            verbose=1
        )
        
        print("Treinamento concluído com sucesso!")
        return model, history
        
    except Exception as e:
        print(f"Erro durante o treinamento: {str(e)}")
        raise

if __name__ == "__main__":
    model, history = main()