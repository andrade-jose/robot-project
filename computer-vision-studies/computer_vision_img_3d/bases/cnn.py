from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, Add, GlobalAveragePooling2D,
    Multiply, Reshape, Activation, ReLU, Concatenate, LayerNormalization
)
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.regularizers import l2
from typing import Optional, Dict

class CNNFactory:
    """Fábrica para criar modelos CNN com arquiteturas flexíveis e otimizadas"""
    
    @staticmethod
    def _se_block(input_tensor, ratio=16):
        """Squeeze-and-Excitation Block para atenção channel-wise"""
        channels = input_tensor.shape[-1]
        
        # Squeeze (global average pooling)
        se = GlobalAveragePooling2D()(input_tensor)
        
        # Excitation (2 FC layers with ReLU and sigmoid)
        se = Dense(channels//ratio, activation='relu', kernel_regularizer=l2(1e-4))(se)
        se = Dense(channels, activation='sigmoid', kernel_regularizer=l2(1e-4))(se)
        
        # Scale the input features
        return Multiply()([input_tensor, se])

    @staticmethod
    def _residual_block(x, filters, kernel_size=3, stride=1):
        """Bloco residual básico com pré-ativação"""
        shortcut = x
        
        # Pré-ativação
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters, kernel_size, strides=stride, padding='same', 
                  kernel_regularizer=l2(1e-4))(x)
        
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters, kernel_size, padding='same', 
                  kernel_regularizer=l2(1e-4))(x)
        
        # Shortcut connection
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = Conv2D(filters, 1, strides=stride, 
                            kernel_regularizer=l2(1e-4))(shortcut)
            shortcut = BatchNormalization()(shortcut)
            
        return Add()([x, shortcut])

    @staticmethod
    def build_basic_cnn(input_shape, num_classes, dropout_rate=0.5):
        """CNN básica com regularização aprimorada"""
        model = Sequential([
            Input(shape=input_shape),
            
            # Bloco 1
            Conv2D(32, 3, padding='same', kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            ReLU(),
            Conv2D(32, 3, padding='same', kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            ReLU(),
            MaxPooling2D(2),
            Dropout(dropout_rate * 0.5),
            
            # Bloco 2
            Conv2D(64, 3, padding='same', kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            ReLU(),
            Conv2D(64, 3, padding='same', kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            ReLU(),
            MaxPooling2D(2),
            Dropout(dropout_rate),

            # Bloco 3
            Conv2D(128, 3, padding='same', kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            ReLU(),
            Conv2D(128, 3, padding='same', kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            ReLU(),
            MaxPooling2D(2),
            Dropout(dropout_rate),
            
            
            # Classificação
            GlobalAveragePooling2D(),
            Dense(512, activation='relu', kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Dropout(dropout_rate),
            Dense(num_classes, activation='softmax', dtype='float32')
        ])
        return model

    @staticmethod
    def build_advanced_cnn(input_shape, num_classes, use_pretrained=False, freeze_backbone=False):
        """CNN avançada com SE Blocks e resíduos"""
        input_layer = Input(shape=input_shape)
        
        if use_pretrained:
            # Backbone pré-treinado
            backbone = EfficientNetB0(
                include_top=False,
                weights='imagenet',
                input_tensor=input_layer
            )
            if freeze_backbone:
                backbone.trainable = False
            x = backbone.output
        else:
            # Camada inicial
            x = Conv2D(64, 3, strides=2, padding='same')(input_layer)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = MaxPooling2D(3, strides=2, padding='same')(x)
            
            # Blocos residuais com SE
            filters_list = [64, 128, 256, 512]
            for i, filters in enumerate(filters_list):
                strides = 1 if i == 0 else 2
                x = CNNFactory._residual_block(x, filters, stride=strides)
                x = CNNFactory._se_block(x)
                x = Dropout(0.2 * (i + 1))(x)
        
        # Topo do modelo
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu', kernel_regularizer=l2(1e-4))(x)
        x = Dropout(0.5)(x)
        output = Dense(num_classes, activation='softmax', dtype='float32')(x)
        
        return Model(inputs=input_layer, outputs=output), backbone if use_pretrained else None


    @staticmethod
    def build_hybrid_model(input_shape, num_classes, use_pretrained=False):
        """Modelo híbrido que processa em paralelo"""
        input_layer = Input(shape=input_shape)
        
        # Processamento compartilhado inicial
        x_shared = Conv2D(64, 3, padding='same')(input_layer)
        x_shared = BatchNormalization()(x_shared)
        x_shared = ReLU()(x_shared)
        x_shared = MaxPooling2D(2)(x_shared)
        
        # Ramificação CNN Básica
        x_basic = Conv2D(128, 3, padding='same')(x_shared)
        x_basic = BatchNormalization()(x_basic)
        x_basic = ReLU()(x_basic)
        x_basic = GlobalAveragePooling2D()(x_basic)
        
        # Ramificação CNN Avançada
        x_advanced = CNNFactory._residual_block(x_shared, 128)
        x_advanced = CNNFactory._se_block(x_advanced)
        x_advanced = GlobalAveragePooling2D()(x_advanced)
        
        # Combinação
        merged = Concatenate()([x_basic, x_advanced])
        merged = Dense(256, activation='relu')(merged)
        output = Dense(num_classes, activation='softmax', dtype='float32')(merged)
        
        return Model(inputs=input_layer, outputs=output)

def create_model(
    architecture: str = 'basic',
    input_shape: tuple = (224, 224, 3),
    num_classes: int = 6,
    use_pretrained: bool = False,
    **kwargs
) -> Model:
    """
    Interface unificada para criação de modelos.
    
    Args:
        architecture: Tipo de arquitetura ('basic', 'advanced', 'hybrid')
        input_shape: Dimensões da imagem de entrada
        num_classes: Número de classes de saída
        use_pretrained: Usar pesos pré-treinados (apenas 'advanced')
        kwargs: Parâmetros adicionais específicos por arquitetura
        
    Returns:
        Modelo Keras compilado
    """
    factory_methods = {
        'basic': CNNFactory.build_basic_cnn,
        'advanced': CNNFactory.build_advanced_cnn,
        'hybrid': CNNFactory.build_hybrid_model
    }
    
    if architecture not in factory_methods:
            raise ValueError(f"Arquitetura inválida. Opções: {list(factory_methods.keys())}")

    if architecture == 'advanced':
        model, backbone = factory_methods[architecture](
            input_shape=input_shape,
            num_classes=num_classes,
            use_pretrained=use_pretrained,
            **kwargs
        )
        return model, backbone
    else:
        model = factory_methods[architecture](
            input_shape=input_shape,
            num_classes=num_classes,
            **kwargs
        )
        return model, None