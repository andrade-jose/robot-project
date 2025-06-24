import os
import sys
import argparse
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import numpy as np
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import TopKCategoricalAccuracy

# Configuração de paths
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

# Imports locais
from architectures.factory import Factory_CNN
from training.callbacks import get_training_callbacks
from data.loading_csv import ShapeDataLoader
from data.preprocessing import create_data_generators
from config.config_advanced import config
from architectures.model_utils import unfreeze_backbone_layers


def parse_args():
    """Analisa argumentos de linha de comando para o modelo advanced"""
    parser = argparse.ArgumentParser(description='Treinar modelo ADVANCED de reconhecimento 3D')
    parser.add_argument('--use_pretrained', action='store_true', help='Usar pesos pré-treinados (EfficientNetB0)')
    parser.add_argument('--freeze_backbone', action='store_true', help='Congelar o backbone pré-treinado')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS, help='Número de épocas de treinamento')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help='Tamanho do batch de treinamento')
    parser.add_argument('--lr', type=float, default=config.INITIAL_LR, help='Taxa de aprendizado inicial')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'], help='Otimizador: sgd ou adam')
    return parser.parse_args()


def setup_environment():
    """Configura o ambiente de treinamento"""
    if tf.config.list_physical_devices('GPU'):
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print('✔️ Mixed precision training habilitado')


def get_optimizer(name, lr):
    """Retorna o otimizador configurado"""
    if name == 'adam':
        return Adam(learning_rate=lr, weight_decay=config.WEIGHT_DECAY)
    return SGD(learning_rate=lr, momentum=0.9, nesterov=True, weight_decay=config.WEIGHT_DECAY)


def main():
    args = parse_args()
    setup_environment()

    # Atualiza configuração
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.INITIAL_LR = args.lr

    architecture = 'advanced'

    print("\n" + "="*50)
    print("🔧 CONFIGURAÇÃO DO TREINAMENTO")
    print(f"Arquitetura: {architecture.upper()}")
    print(f"Pré-treinado: {args.use_pretrained}")
    if args.use_pretrained:
        print(f"Backbone congelado: {args.freeze_backbone}")
    print(f"Épocas: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Otimizador: {args.optimizer.upper()}")
    print("="*50 + "\n")

    os.makedirs(config.TRAINED_MODELS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    try:
        # 1. Dados
        print("[1/5] Carregando dados...")
        data_loader = ShapeDataLoader()
        images, labels = data_loader.load_data_from_csv()
        (x_train, y_train), (x_val, y_val), _, num_classes = data_loader.prepare_datasets(images, labels)

        # 2. Data generators
        print("[2/5] Criando geradores...")
        train_gen, val_gen = create_data_generators(x_train, y_train, x_val, y_val)

        # 3. Modelo
        print("[3/5] Criando modelo ADVANCED...")
        model, backbone = Factory_CNN.create_model(
            architecture='advanced',
            input_shape=config.IMG_SHAPE,
            num_classes=num_classes,
            use_pretrained=args.use_pretrained,
            freeze_backbone=args.freeze_backbone
        )

        # 4. Compilação
        print("[4/5] Compilando modelo...")
        optimizer = get_optimizer(args.optimizer, args.lr)
        model.compile(
            optimizer=optimizer,
            loss=CategoricalCrossentropy(),
            metrics=[
                'accuracy',
                TopKCategoricalAccuracy(k=3, name='top3_accuracy'),
                TopKCategoricalAccuracy(k=5, name='top5_accuracy')
            ]
        )

        # 5. Treinamento
        print("[5/5] Iniciando treinamento...\n")
        history = model.fit(
            train_gen,
            epochs=args.epochs,
            validation_data=val_gen,
            callbacks=get_training_callbacks(
                model_type=architecture,
                use_pretrained=args.use_pretrained
            ),
            verbose=1
        )

        # 6. Salvar modelo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{architecture}_{'pretrained' if args.use_pretrained else 'scratch'}_{timestamp}"
        model_path = os.path.join(config.TRAINED_MODELS_DIR, f"{model_name}.h5")
        model.save(model_path)
        print(f"\n✅ Modelo salvo em: {model_path}")

        history_path = os.path.join(config.LOGS_DIR, f"history_{model_name}.npy")
        np.save(history_path, history.history)
        print(f"📈 Histórico salvo em: {history_path}")

        return model, history

    except Exception as e:
        print(f"\n❌ Erro durante o treinamento: {str(e)}")
        raise


if __name__ == "__main__":
    model, history = main()
