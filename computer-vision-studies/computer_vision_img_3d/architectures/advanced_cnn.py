import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Add, Multiply, Dense,
    ReLU, GlobalAveragePooling2D, GlobalAveragePooling1D, TimeDistributed,
    Concatenate, Conv1D, Reshape, Bidirectional, LSTM, GRU,
    Dropout, LayerNormalization, MultiHeadAttention, Activation,
    SeparableConv2D, DepthwiseConv2D, MaxPooling2D, AveragePooling2D,
    Lambda, Permute, RepeatVector, Flatten
)
from keras.saving import register_keras_serializable
import tensorflow.keras.backend as K

@register_keras_serializable()
class ChannelAttention(tf.keras.layers.Layer):
    """Channel Attention Module (CAM)"""
    def __init__(self, reduction_ratio=16, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.fc1 = Dense(max(self.channels // self.reduction_ratio, 8), activation='relu')
        self.fc2 = Dense(self.channels)
        self.gap = GlobalAveragePooling2D()
        self.gmp = Lambda(lambda x: tf.reduce_max(x, axis=[1, 2], keepdims=False))
        super(ChannelAttention, self).build(input_shape)

    def call(self, x):
        # Average pooling path
        avg_pool = self.gap(x)
        avg_pool = self.fc1(avg_pool)
        avg_pool = self.fc2(avg_pool)

        # Max pooling path
        max_pool = self.gmp(x)
        max_pool = self.fc1(max_pool)
        max_pool = self.fc2(max_pool)

        # Combine and apply sigmoid
        attention = tf.nn.sigmoid(avg_pool + max_pool)
        attention = Reshape((1, 1, self.channels))(attention)

        return x * attention

    def get_config(self):
        config = super().get_config()
        config.update({'reduction_ratio': self.reduction_ratio})
        return config

@register_keras_serializable()
class SpatialAttention(tf.keras.layers.Layer):
    """Spatial Attention Module (SAM)"""
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = Conv2D(1, self.kernel_size, padding='same', activation='sigmoid')
        super(SpatialAttention, self).build(input_shape)

    def call(self, x):
        # Average and max pooling along channel dimension
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)

        # Concatenate and apply convolution
        concat = Concatenate(axis=-1)([avg_pool, max_pool])
        attention = self.conv(concat)

        return x * attention

    def get_config(self):
        config = super().get_config()
        config.update({'kernel_size': self.kernel_size})
        return config

@register_keras_serializable()
class CBAM(tf.keras.layers.Layer):
    """Convolutional Block Attention Module"""
    def __init__(self, reduction_ratio=16, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        self.channel_attention = ChannelAttention(self.reduction_ratio)
        self.spatial_attention = SpatialAttention()
        super(CBAM, self).build(input_shape)

    def call(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'reduction_ratio': self.reduction_ratio})
        return config

@register_keras_serializable()
class EnhancedResidualBlock(tf.keras.layers.Layer):
    """Enhanced Residual Block with CBAM and improved design"""
    def __init__(self, filters, stride=1, expansion=4, **kwargs):
        super(EnhancedResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.stride = stride
        self.expansion = expansion

    def build(self, input_shape):
        # Bottleneck architecture
        self.conv1 = Conv2D(self.filters, 1, use_bias=False)
        self.bn1 = BatchNormalization()

        self.conv2 = SeparableConv2D(self.filters, 3, strides=self.stride,
                                    padding='same', use_bias=False)
        self.bn2 = BatchNormalization()

        self.conv3 = Conv2D(self.filters * self.expansion, 1, use_bias=False)
        self.bn3 = BatchNormalization()

        # CBAM attention
        self.cbam = CBAM(reduction_ratio=16)

        # Shortcut connection
        if self.stride != 1 or input_shape[-1] != self.filters * self.expansion:
            self.shortcut = tf.keras.Sequential([
                Conv2D(self.filters * self.expansion, 1, strides=self.stride, use_bias=False),
                BatchNormalization()
            ])
        else:
            self.shortcut = Lambda(lambda x: x)

        self.relu = ReLU()
        super(EnhancedResidualBlock, self).build(input_shape)

    def call(self, x, training=False):
        identity = x

        # Main path
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out, training=training)

        # Apply CBAM attention
        out = self.cbam(out)

        # Shortcut connection
        identity = self.shortcut(identity)

        # Add and activate
        out = Add()([out, identity])
        out = self.relu(out)

        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'stride': self.stride,
            'expansion': self.expansion
        })
        return config

@register_keras_serializable()
class EnhancedRgbViewNet(tf.keras.layers.Layer):
    """Enhanced RGB View Network with modern architecture"""
    def __init__(self, base_filters=64, **kwargs):
        super(EnhancedRgbViewNet, self).__init__(**kwargs)
        self.base_filters = base_filters

    def build(self, input_shape):
        # Stem network
        self.stem = tf.keras.Sequential([
            Conv2D(self.base_filters, 7, strides=2, padding='same', use_bias=False),
            BatchNormalization(),
            ReLU(),
            MaxPooling2D(3, strides=2, padding='same')
        ])

        # Residual blocks
        self.layer1 = self._make_layer(self.base_filters, 2, stride=1)
        self.layer2 = self._make_layer(self.base_filters * 2, 2, stride=2)
        self.layer3 = self._make_layer(self.base_filters * 4, 2, stride=2)
        self.layer4 = self._make_layer(self.base_filters * 8, 2, stride=2)

        # Global processing
        self.gap = GlobalAveragePooling2D()
        self.dropout = Dropout(0.2)

        super(EnhancedRgbViewNet, self).build(input_shape)

    def _make_layer(self, filters, blocks, stride):
        layers = []
        layers.append(EnhancedResidualBlock(filters, stride))
        for _ in range(1, blocks):
            layers.append(EnhancedResidualBlock(filters))
        return tf.keras.Sequential(layers)

    def call(self, x, training=False):
        x = self.stem(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)

        x = self.gap(x)
        x = self.dropout(x, training=training)

        return x

    def compute_output_shape(self, input_shape):
        # The output shape after global average pooling will be (batch_size, channels)
        # For TimeDistributed, we need to return (batch_size, time_steps, features)
        # Since we don't know batch_size, we return (None, features)
        return (input_shape[0], self.base_filters * 8 * 4)  # *4 from expansion in EnhancedResidualBlock

    def get_config(self):
        config = super().get_config()
        config.update({'base_filters': self.base_filters})
        return config

@register_keras_serializable()
class EnhancedDepthViewNet(tf.keras.layers.Layer):
    """Enhanced Depth View Network optimized for depth maps"""
    def __init__(self, base_filters=32, **kwargs):
        super(EnhancedDepthViewNet, self).__init__(**kwargs)
        self.base_filters = base_filters

    def build(self, input_shape):
        # Depth-specific processing
        self.depth_conv1 = Conv2D(self.base_filters, 7, strides=2, padding='same')
        self.depth_bn1 = BatchNormalization()
        self.depth_relu1 = ReLU()

        # Separable convolutions for efficiency
        self.sep_conv1 = SeparableConv2D(self.base_filters * 2, 3, strides=2, padding='same')
        self.sep_bn1 = BatchNormalization()
        self.sep_relu1 = ReLU()

        self.sep_conv2 = SeparableConv2D(self.base_filters * 4, 3, strides=2, padding='same')
        self.sep_bn2 = BatchNormalization()
        self.sep_relu2 = ReLU()

        self.sep_conv3 = SeparableConv2D(self.base_filters * 8, 3, strides=2, padding='same')
        self.sep_bn3 = BatchNormalization()
        self.sep_relu3 = ReLU()

        # Attention for depth features
        self.cbam = CBAM(reduction_ratio=8)

        self.gap = GlobalAveragePooling2D()
        self.dropout = Dropout(0.2)

        super(EnhancedDepthViewNet, self).build(input_shape)

    def call(self, x, training=False):
        x = self.depth_conv1(x)
        x = self.depth_bn1(x, training=training)
        x = self.depth_relu1(x)

        x = self.sep_conv1(x)
        x = self.sep_bn1(x, training=training)
        x = self.sep_relu1(x)

        x = self.sep_conv2(x)
        x = self.sep_bn2(x, training=training)
        x = self.sep_relu2(x)

        x = self.sep_conv3(x)
        x = self.sep_bn3(x, training=training)
        x = self.sep_relu3(x)

        # Apply attention
        x = self.cbam(x)

        x = self.gap(x)
        x = self.dropout(x, training=training)

        return x

    def compute_output_shape(self, input_shape):
        # The output shape after global average pooling will be (batch_size, channels)
        # For TimeDistributed, we need to return (batch_size, time_steps, features)
        return (input_shape[0], self.base_filters * 8)

    def get_config(self):
        config = super().get_config()
        config.update({'base_filters': self.base_filters})
        return config

@register_keras_serializable()
class CrossModalAttention(tf.keras.layers.Layer):
    """Cross-modal attention between RGB and Depth features"""
    def __init__(self, dim, num_heads=8, **kwargs):
        super(CrossModalAttention, self).__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads

    def build(self, input_shape):
        self.attention = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.dim // self.num_heads,
            dropout=0.1
        )
        self.norm = LayerNormalization()
        super(CrossModalAttention, self).build(input_shape)

    def call(self, rgb_features, depth_features, training=False):
        # Cross-attention: RGB queries Depth
        rgb_attended = self.attention(rgb_features, depth_features, training=training)
        rgb_attended = self.norm(rgb_attended + rgb_features)

        # Cross-attention: Depth queries RGB
        depth_attended = self.attention(depth_features, rgb_features, training=training)
        depth_attended = self.norm(depth_attended + depth_features)

        return rgb_attended, depth_attended

    def get_config(self):
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'num_heads': self.num_heads
        })
        return config

@register_keras_serializable()
class TemporalTransformer(tf.keras.layers.Layer):
    """Transformer for temporal modeling across views"""
    def __init__(self, dim, num_heads=8, ff_dim=512, **kwargs):
        super(TemporalTransformer, self).__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim

    def build(self, input_shape):
        self.attention = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.dim // self.num_heads,
            dropout=0.1
        )
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()

        self.ff = tf.keras.Sequential([
            Dense(self.ff_dim, activation='relu'),
            Dropout(0.1),
            Dense(self.dim)
        ])

        super(TemporalTransformer, self).build(input_shape)

    def call(self, x, training=False):
        # Self-attention
        attn_output = self.attention(x, x, training=training)
        x = self.norm1(x + attn_output)

        # Feed-forward
        ff_output = self.ff(x, training=training)
        x = self.norm2(x + ff_output)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim
        })
        return config

class EnhancedMultiviewCNN:
    @staticmethod
    def build_multiview_model(img_size, num_classes, include_aux=False,
                            aux_features_dim=6, base_filters=64):
        """
        Build an enhanced multi-view CNN model with modern architectures.

        Args:
            img_size: Tuple of (height, width) for input images
            num_classes: Number of output classes
            include_aux: Whether to include auxiliary features
            aux_features_dim: Dimension of auxiliary features
            base_filters: Base number of filters for the network
        """
        # Input layers
        rgb_input = Input(shape=(6, img_size[0], img_size[1], 3), name='rgb_input')
        depth_input = Input(shape=(6, img_size[0], img_size[1], 1), name='depth_input')

        inputs = [rgb_input, depth_input]

        if include_aux:
            aux_input = Input(shape=(6, aux_features_dim), name='aux_input')
            inputs.append(aux_input)

        # Enhanced RGB processing
        rgb_net = EnhancedRgbViewNet(base_filters=base_filters)
        x_rgb = TimeDistributed(rgb_net, name='rgb_processing')(rgb_input)

        # Enhanced Depth processing
        depth_net = EnhancedDepthViewNet(base_filters=base_filters//2)
        x_depth = TimeDistributed(depth_net, name='depth_processing')(depth_input)

        # Feature dimension matching
        rgb_feature_dim = base_filters * 8 * 4  # From EnhancedRgbViewNet
        depth_feature_dim = base_filters//2 * 8  # From EnhancedDepthViewNet

        # Project to common dimension - FIXED: Added unique names
        common_dim = 512
        x_rgb = Dense(common_dim, activation='relu', name='rgb_projection')(x_rgb)
        x_depth = Dense(common_dim, activation='relu', name='depth_projection')(x_depth)

        # Cross-modal attention
        cross_attention = CrossModalAttention(dim=common_dim)
        x_rgb_att, x_depth_att = cross_attention(x_rgb, x_depth)

        # Combine RGB and Depth features
        x_combined = Add(name='add_rgb_depth')([x_rgb_att, x_depth_att])
        x_combined = LayerNormalization(name='norm_combined')(x_combined)

        # Auxiliary features processing
        if include_aux:
            # Enhanced auxiliary processing - FIXED: Added unique names
            x_aux = Dense(64, activation='relu', name='aux_dense_1')(aux_input)
            x_aux = LayerNormalization(name='aux_norm_1')(x_aux)
            x_aux = Dense(128, activation='relu', name='aux_dense_2')(x_aux)
            x_aux = Dropout(0.1, name='aux_dropout_1')(x_aux)
            x_aux = Dense(common_dim, activation='relu', name='aux_projection')(x_aux)

            # Fuse auxiliary features
            x_combined = Add(name='add_aux_features')([x_combined, x_aux])
            x_combined = LayerNormalization(name='norm_aux_combined')(x_combined)

        # Temporal Transformer for view sequence modeling
        x = TemporalTransformer(dim=common_dim, num_heads=8)(x_combined)

        # Additional temporal processing with improved RNN
        x = Bidirectional(GRU(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.1), 
                         name='bi_gru_1')(x)
        x = Bidirectional(GRU(128, dropout=0.2, recurrent_dropout=0.1), 
                         name='bi_gru_2')(x)

        # Enhanced classification head - FIXED: All Dense layers now have unique names
        x = Dense(1024, activation='relu', name='classifier_dense_1')(x)
        x = BatchNormalization(name='classifier_bn_1')(x)
        x = Dropout(0.4, name='classifier_dropout_1')(x)

        x = Dense(512, activation='relu', name='classifier_dense_2')(x)
        x = BatchNormalization(name='classifier_bn_2')(x)
        x = Dropout(0.3, name='classifier_dropout_2')(x)

        x = Dense(256, activation='relu', name='classifier_dense_3')(x)
        x = BatchNormalization(name='classifier_bn_3')(x)
        x = Dropout(0.2, name='classifier_dropout_3')(x)

        # Output layer
        output = Dense(num_classes, activation='softmax', name='classification_output')(x)

        # Create model
        model = Model(inputs=inputs, outputs=output, name='EnhancedMultiviewCNN')

        return model

    @staticmethod
    def build_lightweight_model(img_size, num_classes, include_aux=False,
                               aux_features_dim=6, base_filters=32):
        """
        Build a lightweight version for faster training/inference.
        """
        # Similar structure but with reduced parameters
        rgb_input = Input(shape=(6, img_size[0], img_size[1], 3), name='rgb_input')
        depth_input = Input(shape=(6, img_size[0], img_size[1], 1), name='depth_input')

        inputs = [rgb_input, depth_input]

        if include_aux:
            aux_input = Input(shape=(6, aux_features_dim), name='aux_input')
            inputs.append(aux_input)

        # Lightweight RGB processing
        rgb_net = EnhancedRgbViewNet(base_filters=base_filters)
        x_rgb = TimeDistributed(rgb_net, name='rgb_processing')(rgb_input)

        # Lightweight Depth processing
        depth_net = EnhancedDepthViewNet(base_filters=base_filters//2)
        x_depth = TimeDistributed(depth_net, name='depth_processing')(depth_input)

        # Simple feature fusion
        x = Concatenate(axis=-1, name='concat_features')([x_rgb, x_depth])

        # Auxiliary features - FIXED: Added unique names
        if include_aux:
            x_aux = Dense(64, activation='relu', name='lightweight_aux_dense')(aux_input)
            x_aux = GlobalAveragePooling1D(name='lightweight_aux_gap')(x_aux)
            x_aux = RepeatVector(6, name='lightweight_aux_repeat')(x_aux)
            x = Concatenate(axis=-1, name='concat_with_aux')([x, x_aux])

        # Simplified temporal processing
        x = Bidirectional(GRU(128, return_sequences=True, dropout=0.2), 
                         name='lightweight_bi_gru_1')(x)
        x = Bidirectional(GRU(64, dropout=0.2), 
                         name='lightweight_bi_gru_2')(x)

        # Classification head - FIXED: Added unique names
        x = Dense(256, activation='relu', name='lightweight_dense_1')(x)
        x = BatchNormalization(name='lightweight_bn_1')(x)
        x = Dropout(0.3, name='lightweight_dropout_1')(x)

        output = Dense(num_classes, activation='softmax', name='lightweight_classification')(x)

        model = Model(inputs=inputs, outputs=output, name='LightweightMultiviewCNN')

        return model