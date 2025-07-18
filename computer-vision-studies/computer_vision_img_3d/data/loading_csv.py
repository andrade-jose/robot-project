import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence, to_categorical
from config.config_advanced import config
from imgaug import augmenters as iaa
from data.preprocessing import MultiViewAugmentor
import OpenEXR
import Imath

def read_exr_depth(file_path):
    exr_file = OpenEXR.InputFile(file_path)
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    # Lê o canal 'R' — ajuste se seu EXR tiver outro canal
    channel_str = exr_file.channel('R', FLOAT)
    img = np.frombuffer(channel_str, dtype=np.float32)
    img.shape = (height, width)
    return img

class ShapeMultiViewLoader(Sequence):
    def __init__(self, csv_file=None, df=None, batch_size=config.BATCH_SIZE, shuffle=True, augment=True, include_aux=True, **kwargs):
        """
        Loader multiview para TensorFlow/Keras.

        Args:
            csv_file: caminho do CSV com metadados e caminhos das imagens (opcional se df for passado)
            df: DataFrame já carregado e filtrado (opcional, substitui csv_file)
            batch_size: tamanho do batch
            shuffle: embaralhar a ordem dos dados
            augment: aplicar augmentation online (default False)
            include_aux: incluir variáveis auxiliares (ex: cor fundo, material)
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.include_aux = include_aux
        self.augmentor = MultiViewAugmentor

        if df is not None:
            self.df = df
        elif csv_file is not None:
            self.df = pd.read_csv(csv_file)
        else:
            raise ValueError("Você deve fornecer 'csv_file' ou 'df'.")

        self.model_names = self.df['model_name'].unique()
        self.on_epoch_end()


    def __len__(self):
        "Número de batches por época"
        return int(np.ceil(len(self.model_names) / self.batch_size))

    def on_epoch_end(self):
        "Embaralha os modelos se shuffle ativado"
        self.indexes = np.arange(len(self.model_names))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __getitem__(self, index):
        """
        Gera um batch de dados
        """
        # Índices para os modelos do batch
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_model_names = self.model_names[batch_indexes]

        batch_rgb = []
        batch_depth = []
        batch_aux = []
        batch_labels = []

        for model_name in batch_model_names:
            # Filtra as 6 vistas do modelo atual e ordena por view_idx
            model_rows = self.df[self.df['model_name'] == model_name].sort_values('view_idx')

            rgb_views = []
            depth_views = []
            aux_feats = []

            for _, row in model_rows.iterrows():

                img_rgb_path = row['rgb_path']

                # Verifica se o caminho do arquivo RGB existe
                if not os.path.exists(img_rgb_path):
                    raise FileNotFoundError(f"Arquivo RGB não encontrado: {img_rgb_path}")

                img_rgb = cv2.imread(img_rgb_path)
                # Verifica se a imagem RGB foi lida corretamente
                if img_rgb is None:
                    raise ValueError(f"Falha ao ler imagem RGB: {img_rgb_path}")

                # Verifica se a imagem RGB é colorida e converte para RGB se necessário
                img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
                img_rgb = cv2.resize(img_rgb, config.IMG_SIZE)
                img_rgb = img_rgb.astype(np.float32) / 255.0

                 # Verifica se o caminho do arquivo depth existe
                img_depth_path = row['depth_path']
                
                if not os.path.exists(img_depth_path):
                    raise FileNotFoundError(f"❌ Arquivo depth não encontrado: {img_depth_path}")
                
                img_depth = read_exr_depth(img_depth_path)
                if img_depth is None:
                    raise ValueError(f"❌ Falha ao ler imagem depth EXR: {img_depth_path}")

                # Redimensiona para o tamanho esperado
                img_depth = cv2.resize(img_depth, config.IMG_SIZE)

                # Normaliza para [0,1]
                depth_min = img_depth.min()
                depth_max = img_depth.max()
                if depth_max - depth_min < 1e-6:
                    img_depth = np.zeros_like(img_depth)
                else:
                    img_depth = (img_depth - depth_min) / (depth_max - depth_min)

                img_depth = np.expand_dims(img_depth, axis=-1)

                # --- Coleta para batch ---
                rgb_views.append(img_rgb)
                depth_views.append(img_depth)


                if self.include_aux:
                    # Exemplo: concatenar cores de fundo e material como vetor auxiliar
                    bg_color = eval(row['background_color'])  # converte string para lista
                    mat_color = eval(row['material_color'])
                    aux_feats.append(bg_color[:3] + mat_color[:3])  # só RGB, ignora alpha

            # Empilha vistas para formar tensor (6, H, W, C)
            rgb_stack = np.stack(rgb_views)
            depth_stack = np.stack(depth_views)

            # One-hot encoding da label
            label_str = model_rows.iloc[0]['shape_type']
            label_int = config.SHAPE2IDX[label_str]
            one_hot = to_categorical(label_int, num_classes=config.NUM_CLASSES)
            # Se incluir variáveis auxiliares, empilha também
            sample = {
                "rgb": rgb_stack,
                "depth": depth_stack,
                "label": one_hot
            }
            if self.include_aux:
                sample["aux"] = np.array(aux_feats)
            # Aplica augmentations se ativado
            if self.augment:
                sample = self.augmentor().augment_sample(sample)
                

            batch_rgb.append(sample["rgb"])
            batch_depth.append(sample["depth"])
            batch_labels.append(sample["label"])
            if self.include_aux:
                batch_aux.append(sample["aux"])

        # Converte para numpy arrays prontos para o modelo
        batch_rgb = np.array(batch_rgb)           # (batch, 6, H, W, 3)
        batch_depth = np.array(batch_depth)       # (batch, 6, H, W, 1)
        batch_labels = np.array(batch_labels)     # (batch, num_classes)

        if self.include_aux:
            batch_aux = np.array(batch_aux)       # (batch, 6, n_features)

            return {"rgb_input": batch_rgb, "depth_input": batch_depth, "aux_input": batch_aux}, batch_labels
        else:
            return {"rgb_input": batch_rgb, "depth_input": batch_depth}, batch_labels
        
    def train_val_test_split(self, test_size=0.2, val_size=0.1, shuffle=True, random_state=None):
        """
        Divide os dados em conjuntos de treino, validação e teste, preservando agrupamento por modelo.

        Args:
            test_size (float): Proporção do conjunto de teste (ex: 0.2 = 20%)
            val_size (float): Proporção do conjunto de validação (ex: 0.1 = 10%)
            shuffle (bool): Embaralha os modelos antes de dividir
            random_state (int): Semente para reprodutibilidade

        Returns:
            train_df (pd.DataFrame)
            val_df (pd.DataFrame)
            test_df (pd.DataFrame)
        """
        model_names = np.array(self.model_names)
        if shuffle:
            rng = np.random.default_rng(seed=random_state)
            rng.shuffle(model_names)

        n_total = len(model_names)
        n_test = int(n_total * test_size)
        n_val = int(n_total * val_size)
        n_train = n_total - n_test - n_val

        train_names = model_names[:n_train]
        val_names = model_names[n_train:n_train + n_val]
        test_names = model_names[n_train + n_val:]

        train_df = self.df[self.df['model_name'].isin(train_names)].copy()
        val_df = self.df[self.df['model_name'].isin(val_names)].copy()
        test_df = self.df[self.df['model_name'].isin(test_names)].copy()

        return train_df, val_df, test_df
    
    def get_generator(self, df, shuffle=False, augment=False):
        """
        Gera um novo loader com o mesmo batch_size e include_aux, usando o DataFrame fornecido.
        
        Args:
            df (pd.DataFrame): subconjunto filtrado de dados
            shuffle (bool): embaralhar as amostras
            augment (bool): aplicar aumentos de dados
        """
        return ShapeMultiViewLoader(
            df=df,
            batch_size=self.batch_size,
            shuffle=shuffle,
            augment=augment,
            include_aux=self.include_aux
        )

    @classmethod
    def from_dataframe(cls, df, batch_size=32, shuffle=True, augment=False, include_aux=False):
        """
        Cria um ShapeMultiViewLoader a partir de um DataFrame já carregado e filtrado.
        """
        loader = cls(csv_file=None, batch_size=batch_size, shuffle=shuffle, augment=augment, include_aux=include_aux)
        loader.df = df
        loader.model_names = df['model_name'].unique()
        loader.on_epoch_end()
        return loader


