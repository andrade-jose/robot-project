import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Add, Multiply, Dense,
    ReLU, GlobalAveragePooling2D, TimeDistributed,
    Concatenate, Conv1D, Reshape, Bidirectional, LSTM
)


import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, Multiply, Dense, GlobalAveragePooling2D, Reshape

class ResidualSEBlock(tf.keras.layers.Layer):
    def __init__(self, filters, stride=1, reduction_ratio=16, **kwargs):
        super(ResidualSEBlock, self).__init__(**kwargs)
        self.filters = filters
        self.stride = stride
        self.reduction_ratio = reduction_ratio
        
        # Initialize all layers in __init__ (not in call)
        self.conv1 = Conv2D(filters, 3, strides=stride, padding='same')
        self.bn1 = BatchNormalization()
        self.relu1 = ReLU()
        
        self.conv2 = Conv2D(filters, 3, strides=1, padding='same')
        self.bn2 = BatchNormalization()
        
        self.shortcut_conv = Conv2D(filters, 1, strides=stride, padding='same')
        
        # SE components
        self.gap = GlobalAveragePooling2D()
        self.fc1 = Dense(filters // reduction_ratio, activation='relu')
        self.fc2 = Dense(filters, activation='sigmoid')
        self.reshape = Reshape((1, 1, filters))
        
        self.add = Add()
        self.multiply = Multiply()
        self.relu_out = ReLU()

    def call(self, x, training=False):
        shortcut = x
        
        # Main path
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        # Shortcut path
        if shortcut.shape[-1] != self.filters or self.stride != 1:
            shortcut = self.shortcut_conv(shortcut)
        
        # Squeeze-and-Excitation
        se = self.gap(x)
        se = self.fc1(se)
        se = self.fc2(se)
        se = self.reshape(se)
        
        # Combine
        x = self.multiply([x, se])
        x = self.add([x, shortcut])
        x = self.relu_out(x)
        
        return x
    
class RgbViewNet(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.block1 = ResidualSEBlock(64, stride=1)
        self.block2 = ResidualSEBlock(64, stride=1)
        self.gap = GlobalAveragePooling2D()

    def build(self, input_shape):
        # Apenas marca que a camada está construída — as subcamadas cuidarão do resto
        super().build(input_shape)

    def call(self, x, training=False):
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.gap(x)
        return x

    def compute_output_shape(self, input_shape):
        # input_shape = (batch_size, H, W, C)
        return (input_shape[0], 64)  # 64 canais finais após GlobalAveragePooling2D

class Multiview_CNN:
    @staticmethod
    def build_multiview_model(img_size, num_classes, include_aux=False, aux_features_dim=6):
        # Entradas
        rgb_input = Input(shape=(6, img_size[0], img_size[1], 3), name='rgb_input')
        depth_input = Input(shape=(6, img_size[0], img_size[1], 1), name='depth_input')
        if include_aux:
            aux_input = Input(shape=(6, aux_features_dim), name='aux_input')

        # Modelo compartilhado para processar cada vista RGB
        view_model = RgbViewNet()
        x_rgb = TimeDistributed(view_model)(rgb_input)  # (batch, 6, features)

        # Processa mapas de profundidade com Conv1D após reshape
        x_depth = Reshape((6, img_size[0] * img_size[1]))(depth_input)
        x_depth = Conv1D(64, 3, padding='same', activation='relu')(x_depth)

        # Se tiver variáveis auxiliares, concatena também
        if include_aux:
            x_aux = Conv1D(64, 3, padding='same', activation='relu')(aux_input)
            x = Concatenate(axis=-1)([x_rgb, x_depth, x_aux])
        else:
            x = Concatenate(axis=-1)([x_rgb, x_depth])

        # Processa sequência com LSTM
        x = Bidirectional(LSTM(64))(x)
        x = Dense(256, activation='relu')(x)
        output = Dense(num_classes, activation='softmax')(x)

        # Define o modelo final
        if include_aux:
            model = Model(inputs=[rgb_input, depth_input, aux_input], outputs=output)
        else:
            model = Model(inputs=[rgb_input, depth_input], outputs=output)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
