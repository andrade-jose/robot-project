import sys
import logging
from pathlib import Path
import argparse
import cv2

# Ajuste do path do projeto para importar módulos locais
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from stereo_vision.calibration import StereoCalibration
from stereo_vision.utils import list_image_pairs
from config.config import LEFT_DIR, RIGHT_DIR, CHESSBOARD_SIZE, SQUARE_SIZE, CALIB_FILE, RIGHT_TEST_IMG, LEFT_TEST_IMG

def setup_logger():
    logger = logging.getLogger("StereoCalibrationMain")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(levelname)s] %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def validate_setup(logger) -> bool:
    logger.info("Validando configuração dos diretórios e imagens...")

    left_path = Path(LEFT_DIR)
    right_path = Path(RIGHT_DIR)

    if not left_path.exists():
        logger.error(f"Diretório de imagens esquerdas não encontrado: {LEFT_DIR}")
        return False
    if not right_path.exists():
        logger.error(f"Diretório de imagens direitas não encontrado: {RIGHT_DIR}")
        return False

    left_images = list(left_path.glob("*.png")) + list(left_path.glob("*.jpg"))
    right_images = list(right_path.glob("*.png")) + list(right_path.glob("*.jpg"))

    if len(left_images) == 0:
        logger.error(f"Nenhuma imagem encontrada em: {LEFT_DIR}")
        return False
    if len(right_images) == 0:
        logger.error(f"Nenhuma imagem encontrada em: {RIGHT_DIR}")
        return False

    if len(left_images) != len(right_images):
        logger.warning(f"Quantidade diferente de imagens: {len(left_images)} esquerda vs {len(right_images)} direita")

    logger.info(f"Imagens carregadas: {len(left_images)} esquerdas e {len(right_images)} direitas")
    logger.info(f"Tamanho do tabuleiro: {CHESSBOARD_SIZE}, tamanho do quadrado: {SQUARE_SIZE} m")

    return True


def main(max_images: int, logger: logging.Logger):
    logger.info("Iniciando calibração estéreo...")

    calib = StereoCalibration(chessboard_size=CHESSBOARD_SIZE, square_size=SQUARE_SIZE)

    logger.info("Executando calibração completa...")
    success = calib.run_full_calibration(
        left_dir=LEFT_DIR,
        right_dir=RIGHT_DIR,
        max_images=max_images if max_images > 0 else None,
    )

    if not success:
        logger.error("Falha na calibração!")
        return

    logger.info("Validando calibração e calculando erro epipolar médio...")
    calib.validate_calibration()

    logger.info(f"Salvando parâmetros de calibração em: {CALIB_FILE}")
    calib.save_calibration(CALIB_FILE)

    logger.info("Testando carregamento dos parâmetros salvos...")
    calib_test = StereoCalibration(chessboard_size=CHESSBOARD_SIZE, square_size=SQUARE_SIZE)
    calib_test.load_calibration(CALIB_FILE)

    # Teste de retificação com imagens específicas (se fornecidas)
    if LEFT_TEST_IMG and RIGHT_TEST_IMG:
        logger.info("Testando retificação com imagens específicas...")
        try:
            left_img = cv2.imread(LEFT_TEST_IMG)
            right_img = cv2.imread(RIGHT_TEST_IMG)
            
            if left_img is not None and right_img is not None:
                calib_test.test_rectification(left_img, right_img)
            else:
                logger.warning("Não foi possível carregar as imagens de teste específicas")
        except Exception as e:
            logger.warning(f"Erro no teste de retificação específica: {e}")

    # Testa retificação com o primeiro par disponível
    logger.info("Buscando pares de imagens para teste de retificação...")
    pairs = list_image_pairs(LEFT_DIR, RIGHT_DIR)

    if pairs:
        logger.info(f"Encontrados {len(pairs)} pares, testando o primeiro par...")
        try:
            left_img = cv2.imread(pairs[0][0])
            right_img = cv2.imread(pairs[0][1])
            
            if left_img is not None and right_img is not None:
                calib_test.test_rectification(left_img, right_img)
            else:
                logger.warning("Não foi possível carregar o primeiro par de imagens")
        except Exception as e:
            logger.warning(f"Erro no teste de retificação: {e}")
    else:
        logger.warning("Nenhum par de imagem encontrado para teste de retificação.")

    logger.info("=" * 50)
    logger.info("Calibração concluída com sucesso!")
    logger.info("=" * 50)


def test_existing_calibration(logger):
    logger.info("Testando calibração existente...")

    calib = StereoCalibration(chessboard_size=CHESSBOARD_SIZE, square_size=SQUARE_SIZE)
    try:
        calib.load_calibration(CALIB_FILE)
    except FileNotFoundError:
        logger.error(f"Arquivo de calibração não encontrado: {CALIB_FILE}")
        logger.info("Execute a calibração primeiro.")
        return

    pairs = list_image_pairs(LEFT_DIR, RIGHT_DIR)

    if not pairs:
        logger.warning("Nenhum par de imagens encontrado para teste.")
        return

    num_tests = min(3, len(pairs))
    logger.info(f"Testando retificação com {num_tests} pares de imagens...")

    for i, (left_path, right_path) in enumerate(pairs[:num_tests]):
        logger.info(f"Teste {i + 1}: {Path(left_path).name}")
        try:
            left_img = cv2.imread(left_path)
            right_img = cv2.imread(right_path)
            
            if left_img is not None and right_img is not None:
                calib.test_rectification(left_img, right_img)
            else:
                logger.warning(f"Não foi possível carregar as imagens do teste {i + 1}")
        except Exception as e:
            logger.warning(f"Erro durante o teste {i + 1}: {e}")


if __name__ == "__main__":
    logger = setup_logger()

    parser = argparse.ArgumentParser(description="Calibração de câmeras estéreo")
    parser.add_argument("--test", action="store_true", help="Testa calibração existente")
    parser.add_argument("--validate", action="store_true", help="Valida diretórios e imagens")
    parser.add_argument("--max_images", type=int, default=50, help="Número máximo de pares para calibração (0 para ilimitado)")

    args = parser.parse_args()

    if args.validate:
        valid = validate_setup(logger)
        if not valid:
            logger.error("Configuração inválida.")
    elif args.test:
        test_existing_calibration(logger)
    else:
        if validate_setup(logger):
            main(max_images=args.max_images, logger=logger)
        else:
            logger.error("Configuração inválida. Corrija e tente novamente.")