from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

class ShapeRecognizerTrainer:
    def __init__(self, img_shape=(80, 80, 1), output_dim=6):
        self.img_shape = img_shape
        self.output_dim = output_dim
    
    def build_model(self):
        model = Sequential([
            Conv2D(32, (3,3), activation='relu', padding='same', input_shape=self.img_shape),
            BatchNormalization(),
            Conv2D(32, (3,3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2,2)),
            Dropout(0.2),
            
            Conv2D(64, (3,3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3,3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2,2)),
            Dropout(0.3),
            
            Conv2D(128, (3,3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3,3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2,2)),
            Dropout(0.4),
            
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(self.output_dim, activation='softmax')
        ])
        
        optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def train(self, train_gen, val_gen, epochs=50, batch_size=128):
        model = self.build_model()
        
        callbacks = [
            ModelCheckpoint('models/trained/best_model.h5', save_best_only=True),
            EarlyStopping(patience=15, restore_best_weights=True)
        ]
        
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        return model, history