import sys
import logging
from pathlib import Path
import argparse

# Ajuste do path do projeto para importar módulos locais
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from stereo_vision.calibration import StereoCalibration
from stereo_vision.utils import list_image_pairs
from config.config import LEFT_DIR, RIGHT_DIR, CHESSBOARD_SIZE, SQUARE_SIZE, CALIB_FILE, right_img_path, left_img_path


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


def main(min_quality: float, max_images: int, logger: logging.Logger):
    logger.info("Iniciando calibração estéreo...")

    calib = StereoCalibration(chessboard_size=CHESSBOARD_SIZE, square_size=SQUARE_SIZE)

    logger.info("Executando calibração completa...")
    success = calib.run_full_calibration(
        left_dir=LEFT_DIR,
        right_dir=RIGHT_DIR,
        min_quality=min_quality,
        max_images=max_images if max_images > 0 else None,
    )

    if not success:
        logger.error("Falha na calibração!")
        return

    logger.info("Validando calibração e calculando erro epipolar médio...")
    calib._validate_epipolar()  # Método já loga o resultado

    logger.info(f"Salvando parâmetros de calibração em: {CALIB_FILE}")
    calib.save_calibration(CALIB_FILE)

    logger.info("Testando carregamento dos parâmetros salvos...")
    calib_test = StereoCalibration(chessboard_size=CHESSBOARD_SIZE, square_size=SQUARE_SIZE)
    calib_test.load_calibration(CALIB_FILE)

    # Teste de retificação com imagens específicas (se fornecidas)
    if left_img_path and right_img_path:
        logger.info("Testando retificação com imagens específicas...")
        try:
            calib_test.test_rectification(left_img_path, right_img_path)
        except Exception as e:
            logger.warning(f"Erro no teste de retificação específica: {e}")

    # Testa retificação com o primeiro par disponível
    logger.info("Buscando pares de imagens para teste de retificação...")
    pairs = list_image_pairs(LEFT_DIR, RIGHT_DIR)

    if pairs:
        logger.info(f"Encontrados {len(pairs)} pares, testando o primeiro par...")
        try:
            calib_test.test_rectification(pairs[0][0], pairs[0][1])
        except Exception as e:
            logger.warning(f"Erro no teste de retificação: {e}")
    else:
        logger.warning("Nenhum par de imagem encontrado para teste de retificação.")

    logger.info("=" * 50)
    logger.info("Resumo final da calibração:")
    # Você pode incluir mais dados aqui se quiser
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

    for i, (left, right) in enumerate(pairs[:num_tests]):
        logger.info(f"Teste {i + 1}: {Path(left).name}")
        try:
            calib.test_rectification(left, right)
        except Exception as e:
            logger.warning(f"Erro durante o teste {i + 1}: {e}")


if __name__ == "__main__":
    logger = setup_logger()

    parser = argparse.ArgumentParser(description="Calibração de câmeras estéreo")
    parser.add_argument("--test", action="store_true", help="Testa calibração existente")
    parser.add_argument("--validate", action="store_true", help="Valida diretórios e imagens")
    parser.add_argument("--min_quality", type=float, default=0.08, help="Qualidade mínima dos cantos")
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
            main(min_quality=args.min_quality, max_images=args.max_images, logger=logger)
        else:
            logger.error("Configuração inválida. Corrija e tente novamente.")
