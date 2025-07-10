import os
import gc
import caer
import numpy as np
from tensorflow.keras.utils import to_categorical
from config_basic import config

char_path = r'C:\Venv\OpenCv\computer-vision-studies\datasets\shapes_3d'

def prepare_data():
    char_dict = {char: len(os.listdir(os.path.join(char_path, char))) for char in os.listdir(char_path)}
    char_dict = caer.sort_dict(char_dict, descending=True)
    characters = [c[0] for c in char_dict[:config.OUTPUT_DIM]]

    train = caer.preprocess_from_dir(char_path, characters, channels=config.CHANNELS, IMG_SIZE=config.IMG_SIZE, isShuffle=True)
    featureSet, labels = caer.sep_train(train, IMG_SIZE=config.IMG_SIZE)
    del train

    labels = to_categorical(labels, num_classes=config.OUTPUT_DIM)

    x_train, x_val, y_train, y_val = caer.train_val_split(featureSet, labels, val_ratio=0.2)
    del featureSet
    gc.collect()

    return x_train, x_val, y_train, y_val
