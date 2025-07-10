from pathlib import Path

# Diretórios com caminhos absolutos
LEFT_DIR = r"C:\Venv\Robot_project\robot-project\vision-stereo-studies\camera_project\best_img\left"
RIGHT_DIR = r"C:\Venv\Robot_project\robot-project\vision-stereo-studies\camera_project\best_img\right"
OUTPUT_DIR = r"C:\Venv\Robot_project\robot-project\vision-stereo-studies\camera_project\output"

# Arquivo onde será salvo a calibração
CALIB_FILE = r"C:\Venv\Robot_project\robot-project\vision-stereo-studies\camera_project\output\calib.pkl"

# Imagens para teste de retificação
LEFT_TEST_IMG = r"C:\Venv\Robot_project\robot-project\vision-stereo-studies\camera_project\best_img\left\im_L_1.png"
RIGHT_TEST_IMG = r"C:\Venv\Robot_project\robot-project\vision-stereo-studies\camera_project\best_img\right\im_R_1.png"

# Parâmetros do tabuleiro para calibração
CHESSBOARD_SIZE = (11, 7)   # número de cantos internos (colunas, linhas)
SQUARE_SIZE = 0.030        # tamanho do quadrado em metros

# Parâmetros para stereo matching
NUM_DISPARITIES = 16 * 5   # deve ser múltiplo de 16
BLOCK_SIZE = 7             # ímpar, entre 3 e 11

# Visualização para retificação
LINE_SPACING = 40          # espaçamento entre linhas horizontais
LINE_COLOR = (0, 255, 0)   # cor verde para as linhas
