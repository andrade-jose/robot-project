from src.data_processing_a.data_input import prepare_data
from src.training_model_a.networks import build_shape_classifier
from src.training_model_a.schedulers import get_callbacks
from src.utils.config_a import IMG_SHAPE, OUTPUT_DIM, BATCH_SIZE, EPOCHS

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import math

# --- Preparar dados ---
x_train, x_val, y_train, y_val = prepare_data()

# --- Modelo ---
model = build_shape_classifier()

# --- Compilação ---
optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True, weight_decay=1e-4)
model.compile(optimizer=optimizer, loss=CategoricalCrossentropy(), metrics=['accuracy'])

# --- Aumento de dados ---
train_datagen = ImageDataGenerator(
    rotation_range=30, width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
    fill_mode='nearest', rescale=1./255
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
val_gen = val_datagen.flow(x_val, y_val, batch_size=BATCH_SIZE, shuffle=False)

# --- Callbacks ---
callbacks = get_callbacks()

# --- Treinamento ---
history = model.fit(
    train_gen,
    steps_per_epoch=max(1, len(x_train) // BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=val_gen,
    validation_steps=max(1, len(y_val) // BATCH_SIZE),
    callbacks=callbacks,
    verbose=1
)
