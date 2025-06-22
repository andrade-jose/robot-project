from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
import math

def lr_schedule(epoch):
    initial_lr = 0.01
    drop = 0.5
    epochs_drop = 10
    return initial_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))

def get_callbacks():
    return [
        LearningRateScheduler(lr_schedule),
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ModelCheckpoint('best_model_shapes.h5', monitor='val_accuracy', save_best_only=True)
    ]
