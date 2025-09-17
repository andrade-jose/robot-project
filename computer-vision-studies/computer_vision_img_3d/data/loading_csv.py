import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence, to_categorical
import albumentations as A
import OpenEXR
import Imath
from data.preprocessing import MultiViewAugmentor
from config.config_advanced import config

# Função para ler EXR (mapa de profundidade)
def read_exr_depth(file_path):
    exr_file = OpenEXR.InputFile(file_path)
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    channel_str = exr_file.channel('R', FLOAT)
    img = np.frombuffer(channel_str, dtype=np.float32)
    img.shape = (height, width)
    return img

# Augmentor usando albumentations
class AlbumentationsMultiViewAugmentor:
    def __init__(self):
        self.rgb_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20, p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
        ])
        self.depth_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20, p=0.3),
        ])

    def augment_sample(self, sample):
        rgb_views = sample["rgb"]
        depth_views = sample["depth"]
        aug_rgb, aug_depth = [], []

        for i in range(len(rgb_views)):
            rgb = rgb_views[i]
            depth = depth_views[i][:,:,0]  # remove canal para usar albumentations

            rgb_aug = self.rgb_transform(image=rgb)["image"]
            depth_aug = self.depth_transform(image=depth)["image"]
            depth_aug = np.expand_dims(depth_aug, axis=-1)

            aug_rgb.append(rgb_aug)
            aug_depth.append(depth_aug)

        sample["rgb"] = np.array(aug_rgb)
        sample["depth"] = np.array(aug_depth)
        return sample

# Loader de dados multiview
class ShapeMultiViewLoader(Sequence):
    def __init__(self, csv_file=None, df=None, batch_size=32, shuffle=True, augment=True, include_aux=True, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.include_aux = include_aux
        self.augmentor = AlbumentationsMultiViewAugmentor()

        if df is not None:
            self.df = df
        elif csv_file is not None:
            self.df = pd.read_csv(csv_file)
        else:
            raise ValueError("Você deve fornecer 'csv_file' ou 'df'.")

        self.model_names = self.df['model_name'].unique()
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.model_names) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.model_names))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_model_names = self.model_names[batch_indexes]

        batch_rgb, batch_depth, batch_aux, batch_labels = [], [], [], []

        for model_name in batch_model_names:
            model_rows = self.df[self.df['model_name'] == model_name].sort_values('view_idx')
            rgb_views, depth_views, aux_feats = [], [], []

            for _, row in model_rows.iterrows():
                # RGB
                img_rgb_path = row['rgb_path']
                if not os.path.exists(img_rgb_path):
                    raise FileNotFoundError(f"❌ RGB não encontrado: {img_rgb_path}")
                img_rgb = cv2.imread(img_rgb_path)
                if img_rgb is None:
                    raise ValueError(f"❌ Falha ao ler RGB: {img_rgb_path}")
                img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
                img_rgb = cv2.resize(img_rgb, config.IMG_SIZE)
                img_rgb = img_rgb.astype(np.float32) / 255.0

                # Depth
                img_depth_path = row['depth_path']
                if not os.path.exists(img_depth_path):
                    raise FileNotFoundError(f"❌ Depth não encontrado: {img_depth_path}")
                img_depth = read_exr_depth(img_depth_path)
                if img_depth is None:
                    raise ValueError(f"❌ Falha ao ler Depth: {img_depth_path}")
                img_depth = cv2.resize(img_depth, config.IMG_SIZE)
                d_min, d_max = img_depth.min(), img_depth.max()
                if d_max - d_min < 1e-6:
                    img_depth = np.zeros_like(img_depth)
                else:
                    img_depth = (img_depth - d_min) / (d_max - d_min)
                img_depth = np.expand_dims(img_depth, axis=-1)

                rgb_views.append(img_rgb)
                depth_views.append(img_depth)

                if self.include_aux:
                    bg_color = eval(row['background_color'])
                    mat_color = eval(row['material_color'])
                    aux_feats.append(bg_color[:3] + mat_color[:3])

            rgb_stack = np.stack(rgb_views)
            depth_stack = np.stack(depth_views)
            label_str = model_rows.iloc[0]['shape_type']
            label_int = config.SHAPE2IDX[label_str]
            one_hot = to_categorical(label_int, num_classes=config.NUM_CLASSES)

            sample = {"rgb": rgb_stack, "depth": depth_stack, "label": one_hot}
            if self.include_aux:
                sample["aux"] = np.array(aux_feats)

            if self.augment:
                sample = self.augmentor.augment_sample(sample)

            batch_rgb.append(sample["rgb"])
            batch_depth.append(sample["depth"])
            batch_labels.append(sample["label"])
            if self.include_aux:
                batch_aux.append(sample["aux"])

        batch_rgb = np.array(batch_rgb)
        batch_depth = np.array(batch_depth)
        batch_labels = np.array(batch_labels)

        if self.include_aux:
            batch_aux = np.array(batch_aux)
            return {"rgb_input": batch_rgb, "depth_input": batch_depth, "aux_input": batch_aux}, batch_labels
        else:
            return {"rgb_input": batch_rgb, "depth_input": batch_depth}, batch_labels

    def train_val_test_split(self, test_size=0.2, val_size=0.1, shuffle=True, random_state=None):
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
        return ShapeMultiViewLoader(
            df=df,
            batch_size=self.batch_size,
            shuffle=shuffle,
            augment=augment,
            include_aux=self.include_aux
        )

    @classmethod
    def from_dataframe(cls, df, batch_size=32, shuffle=True, augment=False, include_aux=False):
        loader = cls(csv_file=None, df=df, batch_size=batch_size, shuffle=shuffle, augment=augment, include_aux=include_aux)
        loader.on_epoch_end()
        return loader
