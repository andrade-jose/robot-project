# scripts/calibrate.py

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from stereo_vision.calibration import StereoCalibration
from stereo_vision.utils import list_image_pairs
from config.config import LEFT_DIR, RIGHT_DIR, CHESSBOARD_SIZE, SQUARE_SIZE, CALIB_FILE

def main():
    print("[INFO] Iniciando calibração estéreo...")
    calib = StereoCalibration(chessboard_size=CHESSBOARD_SIZE, square_size=SQUARE_SIZE)
    calib.run_full_calibration(str(LEFT_DIR), str(RIGHT_DIR))
    calib.save_calibration(str(CALIB_FILE))

    # Testa um par para visualização
    pairs = list_image_pairs(str(LEFT_DIR), str(RIGHT_DIR))
    if pairs:
        print("[INFO] Mostrando retificação de teste...")
        calib.rectify_pair(pairs[0][0], pairs[0][1])
    else:
        print("[WARNING] Nenhum par de imagem encontrado.")

if __name__ == "__main__":
    main()
