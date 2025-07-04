import os
import sys
import io
import argparse
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import logging
import time
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, AdamW
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import TopKCategoricalAccuracy

# Adiciona a raiz do projeto ao sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Imports locais
from architectures.advanced_cnn import Multiview_CNN
from training.callbacks import get_training_callbacks
from data.loading_csv import ShapeMultiViewLoader
from data.preprocessing import MultiViewAugmentor
from config.config_advanced import config
from config.pose_utils import pose_loss

sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

# Configura o logger global
log_file = os.path.join(config.LOGS_DIR, 'training.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # também imprime no terminal
    ]
)


def parse_args():
    VALID_OPTIMIZERS = ['sgd', 'adam', 'adamw', 'rmsprop']

    parser = argparse.ArgumentParser(description='Treinar modelo ADVANCED de reconhecimento 3D')
    parser.add_argument('--use_pretrained', action='store_true', help='Usar pesos pré-treinados (EfficientNetB0)')
    parser.add_argument('--freeze_backbone', action='store_true', help='Congelar o backbone pré-treinado')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS, help='Número de épocas de treinamento')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help='Tamanho do batch de treinamento')
    parser.add_argument('--lr', type=float, default=config.INITIAL_LR, help='Taxa de aprendizado inicial')
    parser.add_argument(
        '--optimizer',
        type=str,
        default='sgd',
        help=f'Otimizador: escolha entre {VALID_OPTIMIZERS}'
    )

    args = parser.parse_args()
    if args.optimizer.lower() not in VALID_OPTIMIZERS:
        parser.error(f"Otimizador inválido: {args.optimizer}. Escolha entre {VALID_OPTIMIZERS}")

    return args




def setup_environment():
    """Configura o ambiente de treinamento"""
    if tf.config.list_physical_devices('GPU'):
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logging.info('Mixed precision training habilitado')

    # Limita o uso de memória da GPU

def get_optimizer(name: str, lr: float, weight_decay: float = None):
    """
    Retorna um otimizador Keras com os parâmetros fornecidos.

    Args:
        name (str): Nome do otimizador (ex: 'sgd', 'adam', 'adamw', 'rmsprop')
        lr (float): Taxa de aprendizado
        weight_decay (float): Decaimento de peso (L2), se suportado

    Returns:
        Instância do otimizador correspondente
    """
    name = name.lower()
    kwargs = {"learning_rate": lr}
    if weight_decay is not None:
        kwargs["weight_decay"] = weight_decay

    if name == "adam":
        return Adam(**kwargs)
    elif name == "adamw":
        return AdamW(**kwargs)
    elif name == "sgd":
        return SGD(momentum=0.9, nesterov=True, **kwargs)
    elif name == "rmsprop":
        return RMSprop(momentum=0.9, **kwargs)
    else:
        raise ValueError(f"❌ Otimizador '{name}' não suportado. Use: 'adam', 'adamw', 'sgd', 'rmsprop'")


def main():
    args = args
    setup_environment()

    # Atualiza configuração
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.INITIAL_LR = args.lr

    architecture = 'advanced'

    logging.info("\n" + "="*50)
    logging.info("CONFIGURAÇÃO DO TREINAMENTO")
    logging.info(f"Arquitetura: {architecture.upper()}")
    logging.info(f"Pré-treinado: {args.use_pretrained}")
    if args.use_pretrained:
        logging.info(f"Backbone congelado: {args.freeze_backbone}")
    logging.info(f"Épocas: {args.epochs}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Learning rate: {args.lr}")
    logging.info(f"Otimizador: {args.optimizer.upper()}")
    logging.info("="*50 + "\n")

    os.makedirs(config.TRAINED_MODELS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    try:
        # 1. Dados
        logging.info("[1/4] Carregando dados...")
        data_loader = ShapeMultiViewLoader(
        csv_file=config.CSV_FILE,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        augment=True,
        include_aux=True
        )
        # Carrega o DataFrame com os metadados
        train_df, val_df, test_df = data_loader.train_val_test_split(
        test_size=config.TEST_RATIO,
        val_size=config.VAL_RATIO,  
        random_state=config.RANDOM_STATE
        )
        # Gera o DataLoader para o conjunto de treinamento
        train_gen = ShapeMultiViewLoader(
            df=train_df,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            augment=True,
            include_aux=True
        )
        # Gera o DataLoader para o conjunto de validação
        val_gen = ShapeMultiViewLoader(
            df=val_df,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            augment=False,
            include_aux=True
        )
        # Gera o DataLoader para o conjunto de teste
        test_gen = ShapeMultiViewLoader(
            df=test_df,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            augment=False,
            include_aux=True
        )


        # 2. Modelo
        logging.info("[2/4] Criando modelo ADVANCED...")
        model = Multiview_CNN.build_multiview_model(
                img_size=config.IMG_SIZE,
                include_aux=True,
                aux_features_dim=6
        )

        # 4. Compilação
        logging.info("[3/4] Compilando modelo...")
        # Congela o backbone se solicitado
        optimizer = get_optimizer(args.optimizer, args.lr, weight_decay=config.WEIGHT_DECAY)
        model.compile(
            optimizer=optimizer,
            loss= pose_loss,  # Função de perda personalizada
            # Define as métricas de avaliação
            metrics=[translation_error, rotation_error]
        )

        start = time.time()

        # 5. Treinamento
        logging.info("[4/4] Iniciando treinamento...\n")
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

        end = time.time()
        logging.info(f" Tempo total de treinamento: {(end - start)/60:.2f} minutos")

        # Avaliação final no conjunto de teste
        logging.info("🔍 Avaliando no conjunto de teste...")
        test_metrics = model.evaluate(test_gen, verbose=1)

        for name, value in zip(model.metrics_names, test_metrics):
            logging.info(f"{name}: {value:.4f}")
        
        # 6. Salvar modelo e histórico
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{architecture}_{'pretrained' if args.use_pretrained else 'scratch'}_{timestamp}"

        # Caminho do modelo salvo
        model_path = os.path.join(config.TRAINED_MODELS_DIR, f"{model_name}.h5")
        model.save(model_path)
        logging.info(f"\n✅ Modelo salvo em: {model_path}")

        # Caminho do histórico (.npy)
        history_path = os.path.join(config.LOGS_DIR, f"history_{model_name}.npy")
        np.save(history_path, history.history)
        logging.info(f"📦 Histórico (.npy) salvo em: {history_path}")

        # Caminho do histórico (.csv)
        csv_path = os.path.join(config.LOGS_DIR, f"history_{model_name}.csv")
        pd.DataFrame(history.history).to_csv(csv_path, index=False)
        logging.info(f"📊 Histórico (.csv) salvo em: {csv_path}")

        # Log das melhores métricas
        best_val_acc = max(history.history.get("val_accuracy", [0]))
        best_top3 = max(history.history.get("val_top3_accuracy", [0]))
        best_top5 = max(history.history.get("val_top5_accuracy", [0]))

        logging.info(f"\n🏆 Melhores métricas durante o treino:")
        logging.info(f"🔹 val_accuracy: {best_val_acc:.4f}")
        logging.info(f"🔹 val_top3_accuracy: {best_top3:.4f}")
        logging.info(f"🔹 val_top5_accuracy: {best_top5:.4f}")

        return model, history

    except Exception as e:
        logging.info(f"\n Erro durante o treinamento: {str(e)}")
        raise


if __name__ == "__main__":
    model, history = main()
