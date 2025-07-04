import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Add, Multiply, Dense,
    ReLU, GlobalAveragePooling2D, TimeDistributed,
    Concatenate, Conv1D, Reshape, Bidirectional, LSTM
)
from tensorflow.keras.layers import Layer
from config.pose_utils import pose_loss, translation_error, rotation_error


# === Residual Block com Squeeze-and-Excitation ===
class ResidualSEBlock(Layer):
    def __init__(self, filters, stride=1, reduction_ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.stride = stride
        self.reduction_ratio = reduction_ratio

        self.conv1 = Conv2D(filters, 3, strides=stride, padding='same')
        self.bn1 = BatchNormalization()
        self.relu1 = ReLU()

        self.conv2 = Conv2D(filters, 3, strides=1, padding='same')
        self.bn2 = BatchNormalization()
        self.shortcut_conv = Conv2D(filters, 1, strides=stride, padding='same')

        self.gap = GlobalAveragePooling2D()
        self.fc1 = Dense(filters // reduction_ratio, activation='relu')
        self.fc2 = Dense(filters, activation='sigmoid')
        self.reshape = Reshape((1, 1, filters))

        self.add = Add()
        self.multiply = Multiply()
        self.relu_out = ReLU()

    def call(self, x, training=False):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        if shortcut.shape[-1] != self.filters or self.stride != 1:
            shortcut = self.shortcut_conv(shortcut)

        se = self.gap(x)
        se = self.fc1(se)
        se = self.fc2(se)
        se = self.reshape(se)

        x = self.multiply([x, se])
        x = self.add([x, shortcut])
        return self.relu_out(x)


# === Rede para uma única imagem RGB ===
class RgbViewNet(Layer):
    def __init__(self):
        super().__init__()
        self.block1 = ResidualSEBlock(64)
        self.block2 = ResidualSEBlock(64)
        self.gap = GlobalAveragePooling2D()

    def call(self, x, training=False):
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        return self.gap(x)


# === Modelo principal ajustado para regressão de pose ===
class MultiviewPoseNet:
    @staticmethod
    def build_model(img_size=(224, 224), include_aux=False, aux_features_dim=6):
        rgb_input = Input(shape=(6, img_size[0], img_size[1], 3), name='rgb_input')
        depth_input = Input(shape=(6, img_size[0], img_size[1], 1), name='depth_input')
        if include_aux:
            aux_input = Input(shape=(6, aux_features_dim), name='aux_input')

        # Processa cada imagem RGB individualmente
        view_model = RgbViewNet()
        x_rgb = TimeDistributed(view_model)(rgb_input)

        # Processa mapas de profundidade com Conv1D
        x_depth = Reshape((6, img_size[0] * img_size[1]))(depth_input)
        x_depth = Conv1D(64, 3, padding='same', activation='relu')(x_depth)

        if include_aux:
            x_aux = Conv1D(64, 3, padding='same', activation='relu')(aux_input)
            x = Concatenate(axis=-1)([x_rgb, x_depth, x_aux])
            inputs = [rgb_input, depth_input, aux_input]
        else:
            x = Concatenate(axis=-1)([x_rgb, x_depth])
            inputs = [rgb_input, depth_input]

        # LSTM sobre as views
        x = Bidirectional(LSTM(64))(x)
        x = Dense(256, activation='relu')(x)
        output = Dense(7, activation='linear',name='pose_output')(x)  # tx, ty, tz, qx, qy, qz, qw

        model = Model(inputs=inputs, outputs=output)

        return model