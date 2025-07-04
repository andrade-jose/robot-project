# config/config.py

from pathlib import Path

# Caminhos base
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "filtered"
LEFT_DIR = DATA_DIR / "left"
RIGHT_DIR = DATA_DIR / "right"
OUTPUT_DIR = PROJECT_ROOT / "output"
CALIB_FILE = PROJECT_ROOT / "calibration_data" / "calibration.pkl"

# Calibração
CHESSBOARD_SIZE = (8, 5)      # número de cantos internos (cols, rows)
SQUARE_SIZE = 0.026           # em metros

# Stereo Matching
NUM_DISPARITIES = 16 * 5      # múltiplo de 16
BLOCK_SIZE = 7                # ímpar entre 3 e 11

# Visualização
LINE_SPACING = 40             # espaçamento entre linhas horizontais
LINE_COLOR = (0, 255, 0)      # cor verde
