import cv2
import numpy as np
import logging
from config.config import OUTPUT_DIR

class StereoRectify:
    def __init__(self, calib_data):
        """
        calib_data: dicionário com parâmetros da calibração estéreo,
        por exemplo, carregado do pickle com os atributos:
        mtx_left, dist_left, mtx_right, dist_right, R, T, image_size
        """
        self.logger = logging.getLogger("StereoRectify")
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.hasHandlers():
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(ch)

        self.mtx_left = calib_data["mtx_left"]
        self.dist_left = calib_data["dist_left"]
        self.mtx_right = calib_data["mtx_right"]
        self.dist_right = calib_data["dist_right"]
        self.R = calib_data["R"]
        self.T = calib_data["T"]
        self.image_size = calib_data["image_size"]

        self.R1 = None
        self.R2 = None
        self.P1 = None
        self.P2 = None
        self.Q = None

        self.left_map = None
        self.right_map = None

    def compute_rectification(self, alpha=0.0):
        """
        Calcula a retificação estéreo (matrizes R1,R2,P1,P2 e Q) e
        gera mapas de remapeamento para undistort+rectify.
        alpha controla o zoom e bordas:
            -1 = retificação máxima com recorte,
             0 = retificação sem distorção nas bordas (alguma perda),
             1 = mantém toda a imagem, com bordas pretas
        """
        self.logger.info(f"Computando retificação com alpha={alpha}")

        self.R1, self.R2, self.P1, self.P2, self.Q, roi1, roi2 = cv2.stereoRectify(
            self.mtx_left, self.dist_left,
            self.mtx_right, self.dist_right,
            self.image_size,
            self.R, self.T,
            alpha=alpha,
            flags=cv2.CALIB_ZERO_DISPARITY
        )

        self.left_map = cv2.initUndistortRectifyMap(
            self.mtx_left, self.dist_left, self.R1, self.P1,
            self.image_size, cv2.CV_32FC1
        )
        self.right_map = cv2.initUndistortRectifyMap(
            self.mtx_right, self.dist_right, self.R2, self.P2,
            self.image_size, cv2.CV_32FC1
        )

        self.logger.info(f"Retificação calculada. ROI esquerda: {roi1}, ROI direita: {roi2}")

        return roi1, roi2

    def save_maps(self, left_map_path, right_map_path):
        """
        Salva os mapas de remapeamento para uso posterior.
        """
        if self.left_map is None or self.right_map is None:
            self.logger.error("Mapas de retificação não calculados. Execute compute_rectification primeiro.")
            return

        np.savez_compressed(left_map_path, map1=self.left_map[0], map2=self.left_map[1])
        np.savez_compressed(right_map_path, map1=self.right_map[0], map2=self.right_map[1])
        self.logger.info(f"Mapas de retificação salvos em {left_map_path} e {right_map_path}")

    def load_maps(self, left_map_path, right_map_path):
        """
        Carrega mapas de remapeamento salvos para uso.
        """
        left_data = np.load(left_map_path)
        right_data = np.load(right_map_path)
        self.left_map = (left_data['map1'], left_data['map2'])
        self.right_map = (right_data['map1'], right_data['map2'])
        self.logger.info(f"Mapas de retificação carregados de {left_map_path} e {right_map_path}")

    def rectify_pair(self, img_left, img_right, show_result=True):
        """
        Aplica a retificação nas imagens fornecidas (BGR ou grayscale)
        e retorna as imagens retificadas.

        Se show_result=True, mostra as imagens lado a lado com linhas horizontais.
        """
        if self.left_map is None or self.right_map is None:
            self.logger.error("Mapas de retificação não disponíveis. Execute compute_rectification ou load_maps.")
            return None, None

        rectified_left = cv2.remap(img_left, self.left_map[0], self.left_map[1], cv2.INTER_LINEAR)
        rectified_right = cv2.remap(img_right, self.right_map[0], self.right_map[1], cv2.INTER_LINEAR)

        if show_result:
            stacked = np.hstack((rectified_left, rectified_right))
            height = stacked.shape[0]
            line_spacing = 40
            for y in range(0, height, line_spacing):
                cv2.line(stacked, (0, y), (stacked.shape[1], y), (0, 255, 0), 1)

            cv2.imshow("Imagens Retificadas", stacked)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return rectified_left, rectified_right


if __name__ == "__main__":
    import pickle
    import cv2

    # Carregue parâmetros da calibração
    calib_path = OUTPUT_DIR + "/calib.pkl"
    with open(calib_path, "rb") as f:
        calib_data = pickle.load(f)

    rectifier = StereoRectify(calib_data)
    rectifier.compute_rectification(alpha=0.0)

    # Teste com um par de imagens
    left_img_path = calib_data.get("left_test_img", None)
    right_img_path = calib_data.get("right_test_img", None)

    if left_img_path is None or right_img_path is None:
        print("Caminhos das imagens de teste não encontrados no arquivo de calibração.")
    else:
        img_left = cv2.imread(left_img_path)
        img_right = cv2.imread(right_img_path)
        if img_left is None or img_right is None:
            print("Erro ao carregar as imagens de teste.")
        else:
            rectifier.rectify_pair(img_left, img_right)
