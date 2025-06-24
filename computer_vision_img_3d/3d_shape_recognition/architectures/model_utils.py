
from tensorflow.keras.models import Model

def unfreeze_backbone_layers(backbone, until_layer=None):
    """
    Descongela camadas do backbone até uma camada específica.
    Se until_layer for None, descongela todas.
    """
    trainable = False
    for layer in backbone.layers:
        if until_layer and layer.name == until_layer:
            trainable = True
        if trainable:
            layer.trainable = True
