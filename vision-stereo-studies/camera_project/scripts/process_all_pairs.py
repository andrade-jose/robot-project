# scripts/process_all_pairs.py

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from stereo_vision.processor import StereoProcessor
from stereo_vision.utils import list_image_pairs
from config.config import LEFT_DIR, RIGHT_DIR, OUTPUT_DIR, CALIB_FILE

def main():
    print("[INFO] Carregando parâmetros de calibração...")
    processor = StereoProcessor(str(CALIB_FILE))

    print("[INFO] Carregando pares de imagens...")
    pairs = list_image_pairs(str(LEFT_DIR), str(RIGHT_DIR))

    for idx, (left_path, right_path) in enumerate(pairs, 1):
        print(f"[PROCESSANDO] Par {idx}/{len(pairs)}")
        result = processor.process_pair(left_path, right_path)

        base_name = f"pair_{idx:02d}"
        processor.save_results(str(OUTPUT_DIR), base_name, result['disparity'], result['depth_map'])

    print(f"[SUCESSO] Resultados salvos em: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
