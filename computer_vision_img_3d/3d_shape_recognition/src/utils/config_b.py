import os

class Config:
    # Configuração base - não precisa mudar
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Caminhos ABSOLUTOS para seus dados (ajustados para sua estrutura)
    DATA_ROOT = r"C:\Venv\OpenCv\computer-vision-studies\datasets"
    DATASET_NAME = "dataset_20230517_023945"
    
    # Paths completos (não altere estas linhas)
    DATA_DIR = os.path.join(DATA_ROOT, DATASET_NAME)
    IMAGE_DIR = os.path.join(DATA_DIR, "images")
    CSV_FILE = os.path.join(DATA_DIR, "dataset.csv")  # Confirme o nome exato do arquivo

    # Paths de modelos
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    TRAINED_MODELS_DIR = os.path.join(MODELS_DIR, "trained")
    LOGS_DIR = os.path.join(MODELS_DIR, "logs")
    
    # Parâmetros de imagem
    IMG_SIZE = (80, 80)
    CHANNELS = 1
    IMG_SHAPE = (*IMG_SIZE, CHANNELS)
    NUM_CLASSES = 27

    # Parâmetros de treinamento
    BATCH_SIZE = 128
    EPOCHS = 50
    INITIAL_LR = 0.01
    WEIGHT_DECAY = 1e-4
    
    # Divisão de dados
    VAL_RATIO = 0.2
    TEST_RATIO = 0.1
    RANDOM_STATE = 42

config = Config()