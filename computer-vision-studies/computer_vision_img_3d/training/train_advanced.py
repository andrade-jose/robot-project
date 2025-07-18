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
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, 
    TensorBoard, CSVLogger
)

# Adiciona a raiz do projeto ao sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Imports locais - UPDATED TO MATCH NEW ARCHITECTURE
from architectures.advanced_cnn import EnhancedMultiviewCNN
from data.loading_csv import ShapeMultiViewLoader
from data.preprocessing import MultiViewAugmentor
from config.config_advanced import config

sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

# Configura o logger global
log_file = os.path.join(config.LOGS_DIR, 'training.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)


def parse_args():
    """Analisa argumentos de linha de comando para o modelo enhanced"""
    VALID_OPTIMIZERS = ['sgd', 'adam', 'adamw', 'rmsprop']

    parser = argparse.ArgumentParser(description='Treinar modelo ENHANCED de reconhecimento 3D')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS, help='Número de épocas de treinamento')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help='Tamanho do batch de treinamento')
    parser.add_argument('--lr', type=float, default=config.INITIAL_LR, help='Taxa de aprendizado inicial')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Decaimento de peso (L2 regularization)')
    parser.add_argument('--base_filters', type=int, default=64, help='Número base de filtros para a rede')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Fator de suavização de labels')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Taxa de dropout')
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adamw',
        choices=VALID_OPTIMIZERS,
        help=f'Otimizador: escolha entre {VALID_OPTIMIZERS}'
    )
    parser.add_argument('--patience', type=int, default=10, help='Paciência para early stopping')
    parser.add_argument('--lr_patience', type=int, default=5, help='Paciência para redução de learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-7, help='Learning rate mínimo')
    parser.add_argument('--use_tensorboard', action='store_true', help='Usar TensorBoard para monitoramento')
    parser.add_argument('--resume_training', type=str, default=None, help='Caminho do modelo para continuar treinamento')
    parser.add_argument('--model_type', type=str, default='full', choices=['full', 'lightweight'], 
                       help='Tipo de modelo: full ou lightweight')

    return parser.parse_args()


def setup_environment():
    """Configura o ambiente de treinamento"""
    # Configuração de GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Configura crescimento de memória
            for gpu in gpus:
                tf.config.set_logical_device_configuration(gpu, True)
            
            # Habilita mixed precision
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logging.info(f'✅ GPU disponível: {len(gpus)} dispositivo(s)')
            logging.info('✅ Mixed precision training habilitado')
        except RuntimeError as e:
            logging.error(f'❌ Erro na configuração de GPU: {e}')
    else:
        logging.warning('⚠️ Nenhuma GPU disponível, usando CPU')


def get_optimizer(name: str, lr: float, weight_decay: float = None):
    """
    Retorna um otimizador Keras com os parâmetros fornecidos.
    
    Args:
        name (str): Nome do otimizador
        lr (float): Taxa de aprendizado
        weight_decay (float): Decaimento de peso
    """
    name = name.lower()
    kwargs = {"learning_rate": lr}
    
    if weight_decay is not None and weight_decay > 0:
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


def setup_model_compilation(model, optimizer_name, learning_rate, weight_decay, 
                          label_smoothing, num_classes):
    """Compilação melhorada do modelo com loss e métricas otimizadas"""
    
    # Loss com label smoothing para melhor generalização
    loss = CategoricalCrossentropy(
        label_smoothing=label_smoothing,
        from_logits=False
    )
    
    # Métricas mais abrangentes
    metrics = [
        'accuracy',
        TopKCategoricalAccuracy(k=3, name='top3_accuracy'),
        TopKCategoricalAccuracy(k=5, name='top5_accuracy'),
        'categorical_crossentropy'
    ]
    
    # Otimizador com weight decay
    optimizer = get_optimizer(optimizer_name, learning_rate, weight_decay)
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    logging.info(f'✅ Modelo compilado com {optimizer_name.upper()}')
    logging.info(f'   - Learning rate: {learning_rate}')
    logging.info(f'   - Weight decay: {weight_decay}')
    logging.info(f'   - Label smoothing: {label_smoothing}')
    
    return model


def get_enhanced_callbacks(model_name, patience=10, lr_patience=5, min_lr=1e-7, 
                         use_tensorboard=False):
    """Callbacks melhorados para treinamento robusto"""
    
    callbacks = []
    
    # Diretórios para salvar callbacks
    os.makedirs(config.TRAINED_MODELS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    
    # 1. Redução de learning rate
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=lr_patience,
        min_lr=min_lr,
        verbose=1,
        cooldown=2
    )
    callbacks.append(lr_scheduler)
    
    # 2. Early stopping melhorado
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=patience,
        restore_best_weights=True,
        verbose=1,
        mode='max'
    )
    callbacks.append(early_stopping)
    
    # 3. Model checkpoint
    checkpoint_path = os.path.join(config.TRAINED_MODELS_DIR, f'best_{model_name}.h5')
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
        mode='max'
    )
    callbacks.append(checkpoint)
    
    # 4. CSV Logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_logger = CSVLogger(
    os.path.join(config.LOGS_DIR, f'training_{model_name}_{timestamp}.csv'),
    append=False  # ou True, se você estiver continuando o mesmo log
    )
    callbacks.append(csv_logger)
    
    # 5. TensorBoard (opcional)
    if use_tensorboard:
        tensorboard_dir = os.path.join(config.LOGS_DIR, 'tensorboard', model_name)
        os.makedirs(tensorboard_dir, exist_ok=True)
        tensorboard = TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        )
        callbacks.append(tensorboard)
        logging.info(f'📊 TensorBoard logs: {tensorboard_dir}')
    
    return callbacks


def create_data_generators(batch_size, include_aux=True):
    """Cria os geradores de dados para treino, validação e teste"""
    
    logging.info("📁 Carregando dados...")
    
    # Cria o loader base com o CSV
    base_loader = ShapeMultiViewLoader(
        csv_file=config.CSV_FILE,
        batch_size=batch_size,
        shuffle=True,
        augment=True,
        include_aux=include_aux
    )

    # Divide os dados em conjuntos (agrupado por modelo)
    train_df, val_df, test_df = base_loader.train_val_test_split(
        test_size=config.TEST_RATIO,
        val_size=config.VAL_RATIO,
        random_state=config.RANDOM_STATE
    )

    # Usa o loader base para criar os geradores
    train_gen = base_loader.get_generator(train_df, shuffle=True, augment=True)
    val_gen   = base_loader.get_generator(val_df, shuffle=False, augment=False)
    test_gen  = base_loader.get_generator(test_df, shuffle=False, augment=False)

        
    return train_gen, val_gen, test_gen


def create_model(img_size, num_classes, base_filters, include_aux=True, 
                aux_features_dim=6, model_type='full'):
    """Cria o modelo enhanced - UPDATED TO MATCH NEW ARCHITECTURE"""
    
    logging.info("🏗️ Criando modelo Enhanced...")
    
    # Choose between full and lightweight model
    if model_type == 'lightweight':
        model = EnhancedMultiviewCNN.build_lightweight_model(
            img_size=img_size,
            num_classes=num_classes,
            include_aux=include_aux,
            aux_features_dim=aux_features_dim,
            base_filters=base_filters
        )
        logging.info("📱 Usando modelo lightweight")
    else:
        model = EnhancedMultiviewCNN.build_multiview_model(
            img_size=img_size,
            num_classes=num_classes,
            include_aux=include_aux,
            aux_features_dim=aux_features_dim,
            base_filters=base_filters
        )
        logging.info("🚀 Usando modelo full")
    
    # Mostra resumo do modelo
    model.summary()
    
    # Calcula parâmetros
    total_params = model.count_params()
    logging.info(f"📊 Total de parâmetros: {total_params:,}")
    
    return model


def evaluate_model(model, test_gen, model_name):
    """Avalia o modelo no conjunto de teste"""
    
    logging.info("🔍 Avaliando modelo no conjunto de teste...")
    
    # Avaliação detalhada
    test_metrics = model.evaluate(test_gen, verbose=1)
    
    # Log das métricas
    logging.info("\n📈 MÉTRICAS FINAIS NO TESTE:")
    for name, value in zip(model.metrics_names, test_metrics):
        logging.info(f"   {name}: {value:.4f}")
    
    # Salva métricas em arquivo
    metrics_dict = dict(zip(model.metrics_names, test_metrics))
    metrics_df = pd.DataFrame([metrics_dict])
    metrics_path = os.path.join(config.LOGS_DIR, f'test_metrics_{model_name}.csv')
    metrics_df.to_csv(metrics_path, index=False)
    
    logging.info(f"💾 Métricas salvas em: {metrics_path}")
    
    return test_metrics


def save_model_and_history(model, history, model_name):
    """Salva o modelo e histórico de treinamento"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_model_name = f"{model_name}_{timestamp}"
    
    # Salva o modelo
    model_path = os.path.join(config.TRAINED_MODELS_DIR, f"{full_model_name}.keras")
    model.save(model_path)
    logging.info(f"✅ Modelo salvo em: {model_path}")
    
    # Salva histórico (.npy)
    history_path = os.path.join(config.LOGS_DIR, f"history_{full_model_name}.npy")
    np.save(history_path, history.history)
    logging.info(f"📦 Histórico (.npy) salvo em: {history_path}")
    
    # Salva histórico (.csv)
    csv_path = os.path.join(config.LOGS_DIR, f"history_{full_model_name}.csv")
    pd.DataFrame(history.history).to_csv(csv_path, index=False)
    logging.info(f"📊 Histórico (.csv) salvo em: {csv_path}")
    
    return model_path, full_model_name


def log_training_summary(history, training_time):
    """Registra resumo do treinamento"""
    
    # Melhores métricas
    best_val_acc = max(history.history.get("val_accuracy", [0]))
    best_val_loss = min(history.history.get("val_loss", [float('inf')]))
    best_top3 = max(history.history.get("val_top3_accuracy", [0]))
    best_top5 = max(history.history.get("val_top5_accuracy", [0]))
    
    logging.info("\n" + "="*60)
    logging.info("🏆 RESUMO DO TREINAMENTO")
    logging.info("="*60)
    logging.info(f"⏱️  Tempo total: {training_time:.2f} minutos")
    logging.info(f"🎯 Melhor val_accuracy: {best_val_acc:.4f}")
    logging.info(f"📉 Melhor val_loss: {best_val_loss:.4f}")
    logging.info(f"🥉 Melhor top3_accuracy: {best_top3:.4f}")
    logging.info(f"🏅 Melhor top5_accuracy: {best_top5:.4f}")
    logging.info("="*60)


def main():
    """Função principal do treinamento"""
    
    # Parse argumentos
    args = parse_args()
    
    # Setup ambiente
    setup_environment()
    
    # Atualiza configuração com argumentos
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.INITIAL_LR = args.lr
    
    # Nome do modelo - UPDATED TO REFLECT NEW ARCHITECTURE
    model_name = f"enhanced_{args.model_type}_{args.optimizer}_lr{args.lr}_wd{args.weight_decay}_bf{args.base_filters}"
    
    # Log da configuração
    logging.info("\n" + "="*60)
    logging.info("🚀 CONFIGURAÇÃO DO TREINAMENTO")
    logging.info("="*60)
    logging.info(f"🏗️  Arquitetura: ENHANCED MULTIVIEW CNN")
    logging.info(f"🎯 Tipo de modelo: {args.model_type.upper()}")
    logging.info(f"📊 Épocas: {args.epochs}")
    logging.info(f"📦 Batch size: {args.batch_size}")
    logging.info(f"📈 Learning rate: {args.lr}")
    logging.info(f"⚖️  Weight decay: {args.weight_decay}")
    logging.info(f"🔧 Otimizador: {args.optimizer.upper()}")
    logging.info(f"🎯 Base filters: {args.base_filters}")
    logging.info(f"🎭 Label smoothing: {args.label_smoothing}")
    logging.info(f"💧 Dropout rate: {args.dropout_rate}")
    logging.info(f"⏳ Patience: {args.patience}")
    logging.info(f"📊 TensorBoard: {'Sim' if args.use_tensorboard else 'Não'}")
    logging.info("="*60 + "\n")
    
    # Cria diretórios
    os.makedirs(config.TRAINED_MODELS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    
    try:
        # 1. Carrega dados
        train_gen, val_gen, test_gen = create_data_generators(
            batch_size=args.batch_size,
            include_aux=True
        )
        
        # 2. Cria modelo - UPDATED TO PASS MODEL_TYPE
        model = create_model(
            img_size=config.IMG_SIZE,
            num_classes=config.NUM_CLASSES,
            base_filters=args.base_filters,
            include_aux=True,
            aux_features_dim=6,
            model_type=args.model_type
        )
        
        # 3. Carrega modelo pré-treinado se especificado
        if args.resume_training:
            logging.info(f"🔄 Carregando modelo: {args.resume_training}")
            model = tf.keras.models.load_model(args.resume_training)
        
        # 4. Compila modelo
        model = setup_model_compilation(
            model=model,
            optimizer_name=args.optimizer,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            num_classes=config.NUM_CLASSES
        )
        
        # 5. Callbacks
        callbacks = get_enhanced_callbacks(
            model_name=model_name,
            patience=args.patience,
            lr_patience=args.lr_patience,
            min_lr=args.min_lr,
            use_tensorboard=args.use_tensorboard
        )
        
        # 6. Treinamento
        logging.info("🎯 Iniciando treinamento...\n")
        start_time = time.time()
        
        history = None
        try:
            history = model.fit(
                train_gen,
                epochs=args.epochs,
                validation_data=val_gen,
                callbacks=callbacks,
                verbose=1
            )
        finally:
            if history:
                save_model_and_history(model, history, model_name)
            else:
                logging.warning("⚠️ Treinamento interrompido antes de salvar o histórico.")

        
        end_time = time.time()
        training_time = (end_time - start_time) / 60
        
        # 7. Avaliação final
        test_metrics = evaluate_model(model, test_gen, model_name)
        
        # 8. Salva modelo e histórico
        model_path, full_model_name = save_model_and_history(model, history, model_name)
        
        # 9. Log do resumo
        log_training_summary(history, training_time)
        
        logging.info(f"\n🎉 Treinamento concluído com sucesso!")
        logging.info(f"📁 Modelo final: {model_path}")
        
        return model, history, test_metrics
        
    except Exception as e:
        logging.error(f"\n❌ Erro durante o treinamento: {str(e)}")
        raise
    
    except KeyboardInterrupt:
        logging.info("\n⚠️ Treinamento interrompido pelo usuário")
        return None, None, None


if __name__ == "__main__":
    model, history, test_metrics = main()