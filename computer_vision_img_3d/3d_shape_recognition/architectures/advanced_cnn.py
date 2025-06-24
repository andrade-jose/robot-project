from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, Add, GlobalAveragePooling2D,
    Multiply, Activation, ReLU
)
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.regularizers import l2


class Advanced_CNN:
    """Fábrica para criar modelos CNN com arquiteturas flexíveis e otimizadas"""

    @staticmethod
    def _se_block(input_tensor, ratio=16):
        channels = input_tensor.shape[-1]
        se = GlobalAveragePooling2D()(input_tensor)
        se = Dense(channels // ratio, activation='relu', kernel_regularizer=l2(1e-4))(se)
        se = Dense(channels, activation='sigmoid', kernel_regularizer=l2(1e-4))(se)
        return Multiply()([input_tensor, se])

    @staticmethod
    def _residual_block(x, filters, kernel_size=3, stride=1):
        shortcut = x
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters, kernel_size, strides=stride, padding='same', kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters, kernel_size, padding='same', kernel_regularizer=l2(1e-4))(x)

        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = Conv2D(filters, 1, strides=stride, kernel_regularizer=l2(1e-4))(shortcut)
            shortcut = BatchNormalization()(shortcut)

        return Add()([x, shortcut])

    @staticmethod
    def build_advanced_cnn(input_shape, num_classes, use_pretrained=False, freeze_backbone=False):
        input_layer = Input(shape=input_shape)

        if use_pretrained:
            backbone = EfficientNetB0(
                include_top=False,
                weights='imagenet',
                input_tensor=input_layer
            )
            if freeze_backbone:
                backbone.trainable = False
            x = backbone.output
        else:
            x = Conv2D(64, 3, strides=2, padding='same')(input_layer)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = MaxPooling2D(3, strides=2, padding='same')(x)

            filters_list = [64, 128, 256, 512]
            for i, filters in enumerate(filters_list):
                strides = 1 if i == 0 else 2
                x = Advanced_CNN._residual_block(x, filters, stride=strides)
                x = Advanced_CNN._se_block(x)
                x = Dropout(0.2 * (i + 1))(x)

            backbone = None

        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu', kernel_regularizer=l2(1e-4))(x)
        x = Dropout(0.5)(x)
        output = Dense(num_classes, activation='softmax', dtype='float32')(x)

        return Model(inputs=input_layer, outputs=output), backbone
