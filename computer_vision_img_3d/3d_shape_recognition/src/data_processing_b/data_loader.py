import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from pathlib import Path
from src.utils.config_b import config

class ShapeDataLoader:
    def __init__(self):
        self.csv_file = config.CSV_FILE
        self.image_dir = config.IMAGE_DIR
        
        # Verificação extensiva
        print("\n" + "="*50)
        print("Verificando caminhos de dados:")
        print(f"Diretório base: {config.DATA_DIR}")
        print(f"Pasta de imagens: {self.image_dir}")
        print(f"Arquivo CSV: {self.csv_file}")
        
        if not os.path.exists(self.image_dir):
            print(f"\nERRO: Pasta de imagens não encontrada!")
            print(f"O sistema procurou em: {self.image_dir}")
            print(f"Conteúdo do diretório pai: {os.listdir(config.DATA_DIR)}")
            raise FileNotFoundError(f"Pasta de imagens não encontrada: {self.image_dir}")
        
        if not os.path.exists(self.csv_file):
            print(f"\nERRO: Arquivo CSV não encontrado!")
            print(f"O sistema procurou em: {self.csv_file}")
            print(f"Arquivos disponíveis: {os.listdir(config.DATA_DIR)}")
            raise FileNotFoundError(f"Arquivo CSV não encontrado: {self.csv_file}")
        
        print("Todos os caminhos foram verificados com sucesso!")
        print("="*50 + "\n")

    def load_data_from_csv(self):
        """Carrega imagens e rótulos a partir do CSV com verificações robustas"""
        try:
            df = pd.read_csv(self.csv_file)
            
            # Verificação das colunas necessárias
            if 'filename' not in df.columns or 'shape' not in df.columns:
                missing = [col for col in ['filename', 'shape'] if col not in df.columns]
                raise ValueError(f"Coluna(s) obrigatória(s) não encontrada(s): {missing}")
            
            images = []
            labels = []
            missing_images = []
            
            for idx, row in df.iterrows():
                img_path = os.path.join(self.image_dir, row['filename'])
                
                try:
                    img = self._load_and_preprocess_image(img_path)
                    images.append(img)
                    labels.append(row['shape'])
                except Exception as e:
                    print(f"Erro ao processar {row['filename']}: {str(e)}")
                    missing_images.append(row['filename'])
                    continue
            
            if not images:
                raise ValueError("Nenhuma imagem foi carregada com sucesso")
            
            if missing_images:
                print(f"\nAviso: {len(missing_images)} imagens não puderam ser carregadas")
                print("Exemplo de arquivos faltantes:", missing_images[:5])
            
            # Verificação de classes
            labels = np.array(labels)
            unique_classes = np.unique(labels)
            num_classes = len(unique_classes)
            
            print("\nResumo do carregamento:")
            print(f"- Total de imagens carregadas: {len(images)}")
            print(f"- Classes detectadas: {unique_classes}")
            print(f"- Número de classes: {num_classes}")
            print(f"- Distribuição de classes: {dict(zip(*np.unique(labels, return_counts=True)))}")
            
            if num_classes > config.NUM_CLASSES:
                raise ValueError(
                    f"Dataset contém {num_classes} classes, mas config.NUM_CLASSES = {config.NUM_CLASSES}\n"
                    f"Classes encontradas: {unique_classes}"
                )
            
            return np.array(images), labels
        
        except Exception as e:
            print("\nErro durante o carregamento dos dados:")
            print(f"Tipo do erro: {type(e).__name__}")
            print(f"Detalhes: {str(e)}")
            
            # Informações adicionais para debug
            if 'df' in locals():
                print("\nInformações do DataFrame:")
                print(f"- Total de entradas: {len(df)}")
                print(f"- Colunas disponíveis: {list(df.columns)}")
                print(f"- Primeiros rótulos: {df['shape'].head() if 'shape' in df.columns else 'N/A'}")
            
            raise

    def _load_and_preprocess_image(self, img_path):
        """Carrega e pré-processa uma única imagem"""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Não foi possível carregar a imagem: {img_path}")

        # Conversão de cor
        if config.CHANNELS == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Redimensionamento
        img = cv2.resize(img, config.IMG_SIZE)
        img = img.astype('float32') / 255.0

        # Ajuste do shape para grayscale
        if config.CHANNELS == 1:
            img = np.expand_dims(img, axis=-1)

        return img

    def prepare_datasets(self, images, labels):
        """Divide os dados em treino, validação e teste com verificação robusta"""
        # Verificação do número de classes
        unique_classes = np.unique(labels)
        detected_num_classes = len(unique_classes)
        
        print(f"\nClasses detectadas: {unique_classes}")
        print(f"Número de classes detectado: {detected_num_classes}")
        print(f"Número de classes configurado: {config.NUM_CLASSES}")
        
        # Verificação de consistência
        if detected_num_classes > config.NUM_CLASSES:
            raise ValueError(f"Dataset contém {detected_num_classes} classes, mas config.NUM_CLASSES = {config.NUM_CLASSES}")
        
        # One-hot encoding usando o número de classes da configuração
        labels = to_categorical(labels, num_classes=config.NUM_CLASSES)
        
        # Divisão dos dados
        x_train_val, x_test, y_train_val, y_test = train_test_split(
            images, labels,
            test_size=config.TEST_RATIO,
            random_state=config.RANDOM_STATE,
            stratify=labels
        )
        
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val, y_train_val,
            test_size=config.VAL_RATIO/(1-config.TEST_RATIO),
            random_state=config.RANDOM_STATE,
            stratify=y_train_val
        )
        
        print("\nDivisão dos dados concluída com sucesso:")
        print(f"Treino: {len(x_train)} amostras")
        print(f"Validação: {len(x_val)} amostras")
        print(f"Teste: {len(x_test)} amostras")
        
        return (x_train, y_train), (x_val, y_val), (x_test, y_test), config.NUM_CLASSES
