from models.basic_cnn import build_basic_cnn
from models.advanced_cnn import Advanced_CNN

class Factory_CNN:
    @staticmethod
    def create_model(
        architecture: str = 'basic',
        input_shape: tuple = (224, 224, 3),
        num_classes: int = 6,
        use_pretrained: bool = False,
        **kwargs
    ):
        factory_methods = {
            'basic': build_basic_cnn,
            'advanced': Advanced_CNN.build_advanced_cnn,
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
