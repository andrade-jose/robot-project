from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config.config_advanced import config

def get_augmentation_generators():
    """Configura os geradores de aumento de dados"""
    train_datagen = ImageDataGenerator(
        rotation_range=45,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator()
    
    return train_datagen, val_datagen

def create_data_generators(x_train, y_train, x_val, y_val):
    """Cria geradores de dados para treino e validação"""
    train_datagen, val_datagen = get_augmentation_generators()
    
    train_gen = train_datagen.flow(
        x_train,
        y_train,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )
    
    val_gen = val_datagen.flow(
        x_val,
        y_val,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )
    
    return train_gen, val_gen