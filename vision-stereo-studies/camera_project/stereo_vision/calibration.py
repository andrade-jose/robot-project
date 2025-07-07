import cv2
import numpy as np
import glob
import os
import pickle
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from logging.handlers import RotatingFileHandler
from config.config import LEFT_DIR, RIGHT_DIR


class StereoCalibration:
    def __init__(self, chessboard_size=(8, 5), square_size=0.031):
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self._setup_logging()
        self._reset_calibration_data()
        self.logger.info(f"Inicializando calibração para tabuleiro {self.chessboard_size}")

    def _setup_logging(self):
        self.logger = logging.getLogger('StereoCalibration')
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        file_handler = RotatingFileHandler('calibration.log', maxBytes=1_000_000, backupCount=3)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        if not self.logger.hasHandlers():
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def _reset_calibration_data(self):
        self.objpoints = []
        self.imgpoints_left = []
        self.imgpoints_right = []
        self.image_size = None
        self.calibration_flags = (
            cv2.CALIB_RATIONAL_MODEL +
            cv2.CALIB_THIN_PRISM_MODEL +
            cv2.CALIB_FIX_S1_S2_S3_S4
        )

    def load_images(self, left_dir=LEFT_DIR, right_dir=RIGHT_DIR):
        """Carrega imagens dos diretórios, verificando consistência."""
        patterns = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        left_images, right_images = [], []

        for p in patterns:
            left_images.extend(sorted(glob.glob(os.path.join(left_dir, p))))
            right_images.extend(sorted(glob.glob(os.path.join(right_dir, p))))

        if len(left_images) != len(right_images):
            self.logger.error(f"Número diferente de imagens: {len(left_images)} esquerda vs {len(right_images)} direita")
            raise ValueError("Número desigual de imagens entre pastas esquerda e direita")

        self.logger.info(f"Carregados {len(left_images)} pares de imagens")
        return left_images, right_images

    def _enhance_image(self, img):
        """Pré-processa imagem para melhorar detecção de cantos."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Correção gamma adaptativa
        gamma = 1.5
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        corrected = cv2.LUT(gray, table)

        # Filtro bilateral para suavização preservando bordas
        filtered = cv2.bilateralFilter(corrected, 9, 75, 75)

        # CLAHE para contraste local adaptativo
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(filtered)

        return enhanced

    def find_corners(self, images_left, images_right, min_quality=0.15):
        """Detecta cantos do tabuleiro nas imagens, com validação de qualidade."""
        objp = self._prepare_object_points()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

        for idx, (left_path, right_path) in enumerate(zip(images_left, images_right)):
            img_left = cv2.imread(left_path)
            img_right = cv2.imread(right_path)

            if img_left is None or img_right is None:
                self.logger.warning(f"Não foi possível carregar o par {idx} ({left_path}, {right_path})")
                continue

            enhanced_left = self._enhance_image(img_left)
            enhanced_right = self._enhance_image(img_right)

            ret_left, corners_left = self._detect_chessboard(enhanced_left)
            ret_right, corners_right = self._detect_chessboard(enhanced_right)

            if ret_left and ret_right:
                quality = self._calculate_pair_quality(corners_left, corners_right, img_left.shape)
                if quality >= min_quality:
                    self.objpoints.append(objp)
                    self.imgpoints_left.append(corners_left)
                    self.imgpoints_right.append(corners_right)
                    self.logger.info(f"Par {idx} aceito - Qualidade: {quality:.3f}")
                else:
                    self.logger.warning(f"Par {idx} rejeitado - Qualidade baixa: {quality:.3f}")
            else:
                self.logger.warning(f"Par {idx} rejeitado - Cantos não detectados")

        self.logger.info(f"Detecção concluída: {len(self.objpoints)} pares válidos encontrados")

    def _detect_chessboard(self, img):
        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH +
                 cv2.CALIB_CB_NORMALIZE_IMAGE +
                 cv2.CALIB_CB_FAST_CHECK)

        ret, corners = cv2.findChessboardCorners(img, self.chessboard_size, flags)

        if ret:
            corners = cv2.cornerSubPix(img, corners, (15, 15), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001))
            if not self._validate_pattern(corners):
                return False, None

        return ret, corners

    def _validate_pattern(self, corners):
        """Valida padrão geométrico do tabuleiro (convexidade e espaçamento)."""
        corners = corners.reshape(-1, 2)

        hull = cv2.convexHull(corners)
        area_hull = cv2.contourArea(hull)
        area_bbox = (corners.max(0) - corners.min(0)).prod()

        if area_hull < 0.9 * area_bbox:
            return False

        grid = corners.reshape(self.chessboard_size[1], self.chessboard_size[0], 2)
        dx = np.diff(grid, axis=1)
        dy = np.diff(grid, axis=0)

        if (np.std(dx) / np.mean(dx) > 0.2) or (np.std(dy) / np.mean(dy) > 0.2):
            return False

        return True

    def _calculate_pair_quality(self, corners_left, corners_right, image_shape):
        """Calcula qualidade do par de cantos comparando simetria e distribuição."""
        # Pode ser refinado para medir distâncias, simetria, etc.
        # Aqui um exemplo simples baseado em spread dos cantos (exemplo hipotético)
        spread_left = np.ptp(corners_left, axis=0).mean()
        spread_right = np.ptp(corners_right, axis=0).mean()
        quality = min(spread_left, spread_right) / max(spread_left, spread_right)
        return quality

    def calibrate(self):
        if len(self.objpoints) < 10:
            self.logger.error("Número insuficiente de pares válidos para calibração (mínimo 10).")
            raise RuntimeError("Insuficientes pares para calibração.")

        self._calibrate_individual_cameras()
        self._calibrate_stereo_camera()
        self._rectify()
        self._validate()
        self.logger.info("Calibração completa!")

    def _calibrate_individual_cameras(self):
        self.logger.info("Calibrando câmeras individualmente...")
        retL, self.mtx_left, self.dist_left, _, _ = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_left, self.image_size, None, None, flags=self.calibration_flags)
        retR, self.mtx_right, self.dist_right, _, _ = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_right, self.image_size, None, None, flags=self.calibration_flags)
        self.logger.info(f"RMS esquerda: {retL:.4f}, RMS direita: {retR:.4f}")

    def _calibrate_stereo_camera(self):
        self.logger.info("Calibrando sistema estéreo...")
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500, 1e-6)
        flags = (
            cv2.CALIB_USE_INTRINSIC_GUESS +
            cv2.CALIB_FIX_ASPECT_RATIO +
            cv2.CALIB_RATIONAL_MODEL +
            cv2.CALIB_FIX_K3 +
            cv2.CALIB_FIX_TANGENT_DIST
        )
        ret, _, _, _, _, self.R, self.T, self.E, self.F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_left, self.imgpoints_right,
            self.mtx_left, self.dist_left, self.mtx_right, self.dist_right,
            self.image_size, criteria=criteria, flags=flags
        )
        self.logger.info(f"RMS calibração estéreo: {ret:.4f}")
        self._analyze_stereo_geometry()

    def _analyze_stereo_geometry(self):
        baseline = np.linalg.norm(self.T)
        angle_rad = np.linalg.norm(cv2.Rodrigues(self.R)[0])
        angle_deg = np.degrees(angle_rad)
        self.logger.info(f"Baseline (distância entre câmeras): {baseline*1000:.1f} mm")
        self.logger.info(f"Ângulo de rotação relativo: {angle_deg:.2f}°")

        if baseline < 0.05:
            self.logger.warning("Baseline muito pequena, pode comprometer resultados.")
        elif baseline > 0.3:
            self.logger.warning("Baseline muito grande, cuidado com paralaxe excessiva.")

        if angle_deg > 10:
            self.logger.warning("Ângulo de rotação elevado entre câmeras.")

    def _rectify(self, alpha=-1):
        self.logger.info("Calculando mapas de retificação...")
        best_alpha, best_coverage = self._find_best_alpha(alpha)

        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            self.mtx_left, self.dist_left, self.mtx_right, self.dist_right,
            self.image_size, self.R, self.T, alpha=best_alpha,
            flags=cv2.CALIB_ZERO_DISPARITY
        )

        self.left_map = cv2.initUndistortRectifyMap(
            self.mtx_left, self.dist_left, self.R1, self.P1, self.image_size, cv2.CV_32FC1)
        self.right_map = cv2.initUndistortRectifyMap(
            self.mtx_right, self.dist_right, self.R2, self.P2, self.image_size, cv2.CV_32FC1)

        self.logger.info(f"Retificação concluída com alpha={best_alpha:.2f} (cobertura: {best_coverage:.1%})")

    def _find_best_alpha(self, alpha):
        if alpha >= 0:
            _, _, _, _, _, roi1, roi2 = cv2.stereoRectify(
                self.mtx_left, self.dist_left, self.mtx_right, self.dist_right,
                self.image_size, self.R, self.T, alpha=alpha, flags=cv2.CALIB_ZERO_DISPARITY
            )
            coverage = self._calculate_coverage(roi1, roi2)
            return alpha, coverage

        best_alpha = 0.0
        best_coverage = 0.0
        for a in np.linspace(-1, 1, 11):
            _, _, _, _, _, roi1, roi2 = cv2.stereoRectify(
                self.mtx_left, self.dist_left, self.mtx_right, self.dist_right,
                self.image_size, self.R, self.T, alpha=a, flags=cv2.CALIB_ZERO_DISPARITY
            )
            coverage = self._calculate_coverage(roi1, roi2)
            if coverage > best_coverage:
                best_coverage = coverage
                best_alpha = a

        return best_alpha, best_coverage

    def _calculate_coverage(self, roi1, roi2):
        area1 = roi1[2] * roi1[3] if roi1[2] > 0 and roi1[3] > 0 else 0
        area2 = roi2[2] * roi2[3] if roi2[2] > 0 and roi2[3] > 0 else 0
        total_area = self.image_size[0] * self.image_size[1]
        return (area1 + area2) / (2 * total_area)

    def _validate(self):
        self._validate_epipolar()
        self._validate_reprojection()
        self.visualize_results()

    def _validate_epipolar(self):
        mean_error = 0
        total_points = 0

        for ptsL, ptsR in zip(self.imgpoints_left, self.imgpoints_right):
            ptsL = ptsL.reshape(-1, 2)
            ptsR = ptsR.reshape(-1, 2)

            linesR = cv2.computeCorrespondEpilines(ptsL, 1, self.F).reshape(-1, 3)
            linesL = cv2.computeCorrespondEpilines(ptsR, 2, self.F).reshape(-1, 3)

            for ptR, lineR in zip(ptsR, linesR):
                dist = abs(lineR[0]*ptR[0] + lineR[1]*ptR[1] + lineR[2]) / np.linalg.norm(lineR[:2])
                mean_error += dist

            for ptL, lineL in zip(ptsL, linesL):
                dist = abs(lineL[0]*ptL[0] + lineL[1]*ptL[1] + lineL[2]) / np.linalg.norm(lineL[:2])
                mean_error += dist

            total_points += 2 * len(ptsL)

        mean_error /= total_points
        self.logger.info(f"Erro epipolar médio: {mean_error:.4f} pixels")
        if mean_error > 0.5:
            self.logger.warning("Erro epipolar elevado!")

    def _validate_reprojection(self):
        # Método placeholder para reprojeção, pode implementar se desejar
        pass

    def visualize_results(self):
        self._plot_3d_configuration()
        self._plot_rectification_example()

    def _plot_3d_configuration(self):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        self._plot_camera(ax, np.eye(3), np.zeros(3), 'Left Camera')
        self._plot_camera(ax, self.R, self.T.ravel(), 'Right Camera')

        for objp in self.objpoints:
            ax.scatter(objp[:, 0], objp[:, 1], objp[:, 2], c='b', s=10)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        plt.title('Configuração 3D das Câmeras Estéreo')
        plt.tight_layout()
        plt.show()

    def _plot_camera(self, ax, R, t, label):
        axis_length = 0.05
        axes = np.array([[0, 0, 0],
                         [axis_length, 0, 0],
                         [0, axis_length, 0],
                         [0, 0, axis_length]])

        t = t.reshape(3, 1)
        world_axes = (R @ axes.T + t).T

        colors = ['r', 'g', 'b']
        for i in range(1, 4):
            ax.plot([world_axes[0, 0], world_axes[i, 0]],
                    [world_axes[0, 1], world_axes[i, 1]],
                    [world_axes[0, 2], world_axes[i, 2]],
                    colors[i - 1])

        ax.text(world_axes[0, 0], world_axes[0, 1], world_axes[0, 2], label)

    def _prepare_object_points(self):
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        return objp

    def run_full_calibration(self, left_dir=LEFT_DIR, right_dir=RIGHT_DIR, min_quality=0.1, max_images=None):
        left_images, right_images = self.load_images(left_dir, right_dir)

        if max_images:
            left_images = left_images[:max_images]
            right_images = right_images[:max_images]

        self.image_size = cv2.imread(left_images[0]).shape[1::-1]
        self.find_corners(left_images, right_images, min_quality=min_quality)
        self.calibrate()
        return True

    def save_calibration(self, filename):
        data = {
            "mtx_left": self.mtx_left,
            "dist_left": self.dist_left,
            "mtx_right": self.mtx_right,
            "dist_right": self.dist_right,
            "R": self.R,
            "T": self.T,
            "E": self.E,
            "F": self.F,
            "R1": self.R1,
            "R2": self.R2,
            "P1": self.P1,
            "P2": self.P2,
            "Q": self.Q,
            "left_map": self.left_map,
            "right_map": self.right_map,
            "chessboard_size": self.chessboard_size,
            "square_size": self.square_size,
            "image_size": self.image_size,
        }
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        self.logger.info(f"Parâmetros de calibração salvos em {filename}")

    def load_calibration(self, filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        self.mtx_left = data["mtx_left"]
        self.dist_left = data["dist_left"]
        self.mtx_right = data["mtx_right"]
        self.dist_right = data["dist_right"]
        self.R = data["R"]
        self.T = data["T"]
        self.E = data["E"]
        self.F = data["F"]
        self.R1 = data["R1"]
        self.R2 = data["R2"]
        self.P1 = data["P1"]
        self.P2 = data["P2"]
        self.Q = data["Q"]
        self.left_map = data["left_map"]
        self.right_map = data["right_map"]
        self.chessboard_size = data.get("chessboard_size", self.chessboard_size)
        self.square_size = data.get("square_size", self.square_size)
        self.image_size = data.get("image_size", self.image_size)
        self.logger.info(f"Parâmetros de calibração carregados de {filename}")

    def test_rectification(self, left_img_path, right_img_path):
        """Exibe imagem retificada de um par para visualização qualitativa."""
        imgL = cv2.imread(left_img_path)
        imgR = cv2.imread(right_img_path)
        if imgL is None or imgR is None:
            self.logger.error("Imagens para retificação não podem ser carregadas.")
            return

        rectL = cv2.remap(imgL, self.left_map[0], self.left_map[1], cv2.INTER_LINEAR)
        rectR = cv2.remap(imgR, self.right_map[0], self.right_map[1], cv2.INTER_LINEAR)

        stacked = np.hstack((rectL, rectR))
        for y in range(0, stacked.shape[0], 40):
            cv2.line(stacked, (0, y), (stacked.shape[1], y), (0, 255, 0), 1)

        cv2.imshow("Imagens Retificadas - Pressione qualquer tecla para fechar", stacked)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
