import os
import tensorflow as tf
from dataclasses import dataclass

@dataclass
class Config:
    """Classe de configuração para o projeto com organização melhorada"""
    
    # Estrutura de diretórios base (não mude estas linhas)
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Caminhos para os dados (atualize conforme sua estrutura)
    DATA_ROOT: str = r"C:\Venv\OpenCv\computer-vision-studies\datasets"
    DATASET_NAME: str = "dataset_20230517_024012"
    
    # Caminhos completos (calculados automaticamente)
    @property
    def DATA_DIR(self) -> str:
        """Retorna o caminho completo para o diretório de dados"""
        return os.path.join(self.DATA_ROOT, self.DATASET_NAME)
    
    @property
    def IMAGE_DIR(self) -> str:
        """Retorna o caminho para as imagens"""
        return os.path.join(self.DATA_DIR, "images")
    
    @property
    def CSV_FILE(self) -> str:
        """Retorna o caminho para o arquivo CSV"""
        return os.path.join(self.DATA_DIR, "dataset.csv")
    
    # Caminhos para modelos
    @property
    def MODELS_DIR(self) -> str:
        """Diretório base para os modelos"""
        return os.path.join(self.BASE_DIR, "models")
    
    @property
    def TRAINED_MODELS_DIR(self) -> str:
        """Diretório para modelos treinados"""
        return os.path.join(self.MODELS_DIR, "trained")
    
    @property
    def LOGS_DIR(self) -> str:
        """Diretório para logs de treinamento"""
        return os.path.join(self.MODELS_DIR, "logs")
    
    # Parâmetros de imagem
    IMG_SIZE: tuple = (125, 125)          # Tamanho da imagem (altura, largura)
    CHANNELS: int = 3                     # 1 para grayscale, 3 para RGB
    
    @property
    def IMG_SHAPE(self) -> tuple:
        """Retorna a forma completa da imagem (altura, largura, canais)"""
        return (*self.IMG_SIZE, self.CHANNELS)
    
    NUM_CLASSES: int = 27               # Número de classes no seu problema

    # Parâmetros de treinamento
    BATCH_SIZE: int = 32               # Tamanho do lote
    EPOCHS: int = 50                   # Número de épocas
    INITIAL_LR: float = 0.001          # Taxa de aprendizado inicial
    WEIGHT_DECAY: float = 1e-4         # Decaimento de pesos para regularização
    
    # Divisão dos dados
    VAL_RATIO: float = 0.2              # Proporção para validação
    TEST_RATIO: float = 0.1             # Proporção para teste
    RANDOM_STATE: int = 472719           # Semente aleatória para reprodutibilidade
    
    # Configurações de aumento de dados
    ROTATION_RANGE: int = 15            # Rotação aleatória em graus
    WIDTH_SHIFT_RANGE: float = 0.1      # Deslocamento horizontal aleatório
    HEIGHT_SHIFT_RANGE: float = 0.1     # Deslocamento vertical aleatório
    BRIGHTNESS_RANGE: tuple = (0.9, 1.1) # Variação de brilho
    
    # Configurações de hardware
    USE_MIXED_PRECISION: bool = True    # Usar precisão mista se disponível
    GPU_MEMORY_LIMIT: int = 4096        # Limite de memória da GPU em MB

    def __post_init__(self):
        """Configurações iniciais automáticas"""
        # Cria os diretórios se não existirem
        os.makedirs(self.TRAINED_MODELS_DIR, exist_ok=True)
        os.makedirs(self.LOGS_DIR, exist_ok=True)
        
        # Configura a GPU se disponível
        if tf.config.list_physical_devices('GPU') and self.USE_MIXED_PRECISION:
            # Ativa precisão mista para melhor desempenho
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            
            # Configura limite de memória da GPU
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus and self.GPU_MEMORY_LIMIT:
                try:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpus[0],
                        [tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=self.GPU_MEMORY_LIMIT)]
                    )
                except RuntimeError as e:
                    print(f"Erro ao configurar GPU: {e}")

# Inicializa a configuração
config = Config()