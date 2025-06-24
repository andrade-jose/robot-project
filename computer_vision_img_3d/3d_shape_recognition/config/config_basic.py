
class Config:
    IMG_SIZE = (80, 80)
    CHANNELS = 1
    IMG_SHAPE = (80, 80, 1)
    OUTPUT_DIM = 6
    BATCH_SIZE = 128
    EPOCHS = 50
    num_classes = OUTPUT_DIM
    input_shape = IMG_SHAPE
config = Config()