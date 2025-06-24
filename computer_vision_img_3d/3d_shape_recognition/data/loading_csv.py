import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from pathlib import Path
from config.config_advanced import config

class ShapeDataLoader:
    def __init__(self):
        """Inicializa o carregador de dados com verificação robusta dos caminhos"""
        self.csv_file = config.CSV_FILE
        self.image_dir = config.IMAGE_DIR
        
        # Verificação detalhada dos caminhos
        self._verificar_caminhos()
        
    def _verificar_caminhos(self):
        """Realiza verificação completa dos caminhos com diagnósticos detalhados"""
        print("\n" + "="*50)
        print("Verificação de Caminhos de Dados:")
        print(f"Diretório Base: {config.DATA_DIR}")
        print(f"Pasta de Imagens: {self.image_dir}")
        print(f"Arquivo CSV: {self.csv_file}") 
        
        # Verifica pasta de imagens
        if not os.path.exists(self.image_dir):
            print(f"\nERRO: Pasta de imagens não encontrada!")
            print(f"Caminho esperado: {self.image_dir}")
            pasta_pai = str(Path(self.image_dir).parent)
            print(f"Conteúdo do diretório pai: {os.listdir(pasta_pai)}")
            raise FileNotFoundError(f"Pasta de imagens não encontrada: {self.image_dir}")
        
        # Verifica arquivo CSV
        if not os.path.exists(self.csv_file):
            print(f"\nERRO: Arquivo CSV não encontrado!")
            print(f"Caminho esperado: {self.csv_file}")
            print(f"Conteúdo do diretório: {os.listdir(os.path.dirname(self.csv_file))}")
            raise FileNotFoundError(f"Arquivo CSV não encontrado: {self.csv_file}")
        
        print("Todos os caminhos verificados com sucesso!")
        print("="*50 + "\n")

    def carregar_dados(self):
        """Carrega imagens e rótulos do CSV com validação robusta"""
        try:
            # Carrega e valida o CSV
            df = self._carregar_validar_csv()
            
            # Processa imagens e rótulos
            imagens, rotulos = self._processar_imagens_rotulos(df)
            
            # Valida distribuição de classes
            self._validar_distribuicao_classes(rotulos)
            
            return np.array(imagens), np.array(rotulos)
            
        except Exception as e:
            print("\nFalha no carregamento de dados:")
            print(f"Tipo do Erro: {type(e).__name__}")
            print(f"Detalhes: {str(e)}")
            raise

    def load_data_from_csv(self):
        """Carrega imagens e rótulos a partir do CSV"""
        try:
            df = pd.read_csv(self.csv_file)
            
            # Validação (mantenha seu código existente)
            colunas_obrigatorias = ['filename', 'shape']
            colunas_faltantes = [col for col in colunas_obrigatorias if col not in df.columns]
            if colunas_faltantes:
                raise ValueError(f"Colunas obrigatórias faltando: {colunas_faltantes}")
                
            if df['filename'].isnull().any() or df['shape'].isnull().any():
                raise ValueError("CSV contém valores vazios nas colunas obrigatórias")
            
            # Carrega as imagens
            images = []
            labels = []
            
            for _, row in df.iterrows():
                img_path = os.path.join(self.image_dir, row['filename'])
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converte para RGB
                        img = cv2.resize(img, config.IMG_SIZE)  # Redimensiona
                        img = img.astype('float32') / 255.0  # Normaliza
                        images.append(img)
                        labels.append(row['shape'])
                except Exception as e:
                    print(f"Erro ao carregar {row['filename']}: {str(e)}")
                    continue
                    
            if not images:
                raise ValueError("Nenhuma imagem foi carregada com sucesso")
                
            return np.array(images), np.array(labels)
            
        except pd.errors.EmptyDataError:
            raise ValueError("Arquivo CSV está vazio ou corrompido")
        except pd.errors.ParserError:
            raise ValueError("Erro ao ler CSV - verifique o formato do arquivo")

    def _processar_imagens_rotulos(self, df):
        """Processa imagens e rótulos com rastreamento detalhado de erros"""
        imagens = []
        rotulos = []
        arquivos_faltantes = []
        arquivos_corrompidos = []
        
        for idx, row in df.iterrows():
            caminho_imagem = os.path.join(self.image_dir, row['filename'])
            
            try:
                img = self._carregar_preprocessar_imagem(caminho_imagem)
                imagens.append(img)
                rotulos.append(row['shape'])
            except FileNotFoundError:
                arquivos_faltantes.append(row['filename'])
            except Exception as e:
                arquivos_corrompidos.append((row['filename'], str(e)))
                continue
        
        # Gera relatório detalhado de erros
        if arquivos_faltantes or arquivos_corrompidos:
            self._gerar_relatorio_erros(arquivos_faltantes, arquivos_corrompidos, len(df))
            
        if not imagens:
            raise ValueError("Nenhuma imagem foi carregada com sucesso")
            
        return imagens, rotulos

    def _gerar_relatorio_erros(self, arquivos_faltantes, arquivos_corrompidos, total_arquivos):
        """Gera relatório detalhado de erros para debug"""
        taxa_erro = (len(arquivos_faltantes) + len(arquivos_corrompidos)) / total_arquivos * 100
        print(f"\nAviso: {taxa_erro:.2f}% dos arquivos tiveram problemas")
        
        if arquivos_faltantes:
            print(f"- Arquivos faltantes: {len(arquivos_faltantes)}")
            print("  Exemplo de arquivos faltantes:", arquivos_faltantes[:3])
            
        if arquivos_corrompidos:
            print(f"- Arquivos corrompidos: {len(arquivos_corrompidos)}")
            print("  Exemplo de erros:", arquivos_corrompidos[:3])

    def _validar_distribuicao_classes(self, rotulos):
        """Valida distribuição de classes em relação à configuração"""
        rotulos = np.array(rotulos)
        classes_unicas = np.unique(rotulos)
        num_classes = len(classes_unicas)
        
        print("\nResumo do Dataset:")
        print(f"- Total de amostras: {len(rotulos)}")
        print(f"- Classes detectadas: {classes_unicas}")
        print(f"- Distribuição de classes:\n{pd.Series(rotulos).value_counts().to_string()}")
        
        if num_classes != config.NUM_CLASSES:
            raise ValueError(
                f"Dataset contém {num_classes} classes, mas config espera {config.NUM_CLASSES}\n"
                f"Classes detectadas: {sorted(classes_unicas)}\n"
                f"Atualize config.NUM_CLASSES ou verifique seu dataset"
            )

    def _carregar_preprocessar_imagem(self, caminho_imagem):
        """Carrega e pré-processa uma imagem com validação aprimorada"""
        if not os.path.exists(caminho_imagem):
            raise FileNotFoundError(f"Arquivo de imagem não encontrado: {caminho_imagem}")
            
        img = cv2.imread(caminho_imagem)
        if img is None:
            raise ValueError(f"Falha ao carregar imagem (pode estar corrompida): {caminho_imagem}")

        # Conversão de cor
        try:
            if config.CHANNELS == 1:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.expand_dims(img, axis=-1)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise ValueError(f"Falha na conversão de cor para {caminho_imagem}: {str(e)}")

        # Redimensionamento
        try:
            img = cv2.resize(img, config.IMG_SIZE)
            img = img.astype('float32') / 255.0
        except Exception as e:
            raise ValueError(f"Falha no redimensionamento para {caminho_imagem}: {str(e)}")

        return img

    def prepare_datasets(self, imagens, rotulos):
        """Divide os dados em treino/validação/teste com validação robusta"""
        # Verificação de entrada melhorada
        if imagens.size == 0 or rotulos.size == 0:
            raise ValueError("Arrays de entrada vazios")
        
        # Conversão para one-hot encoding
        try:
            rotulos = to_categorical(rotulos, num_classes=config.NUM_CLASSES)
        except Exception as e:
            raise ValueError(f"Erro no one-hot encoding: {str(e)}")

        # Divisão estratificada
        try:
            # Primeira divisão: treino+validação vs teste
            x_treino_val, x_teste, y_treino_val, y_teste = train_test_split(
                imagens, 
                rotulos,
                test_size=config.TEST_RATIO,
                random_state=config.RANDOM_STATE,
                stratify=rotulos
            )
            
            # Segunda divisão: treino vs validação
            x_treino, x_val, y_treino, y_val = train_test_split(
                x_treino_val, 
                y_treino_val,
                test_size=config.VAL_RATIO/(1-config.TEST_RATIO),
                random_state=config.RANDOM_STATE,
                stratify=y_treino_val
            )
        except ValueError as e:
            raise ValueError(f"Erro na divisão dos dados: {str(e)}")

        print("\nDivisão dos dados concluída:")
        print(f"Treino: {len(x_treino)} amostras")
        print(f"Validação: {len(x_val)} amostras")
        print(f"Teste: {len(x_teste)} amostras")
        
        return (x_treino, y_treino), (x_val, y_val), (x_teste, y_teste), config.NUM_CLASSES