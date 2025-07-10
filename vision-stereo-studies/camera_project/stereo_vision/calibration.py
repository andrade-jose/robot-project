import cv2
import numpy as np
import glob
import pickle
import os
import sys
import logging
from pathlib import Path
import argparse

class StereoCalibration:
    def __init__(self, chessboard_size=(11, 7), square_size=0.030):
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.objpoints = []
        self.imgpoints_left = []
        self.imgpoints_right = []
        self.mtx_left = None
        self.dist_left = None
        self.mtx_right = None
        self.dist_right = None
        self.R = None
        self.T = None
        self.E = None
        self.F = None
        self.P1 = None
        self.P2 = None
        self.Q = None
        self.left_map1 = None
        self.left_map2 = None
        self.right_map1 = None
        self.right_map2 = None
        self.image_size = None
        self.roi_left = None
        self.roi_right = None

    def _prepare_object_points(self):
        """Prepara pontos 3D do tabuleiro de xadrez"""
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        print(f"[DEBUG] Pontos do objeto preparados: {objp.shape}")
        print(f"[DEBUG] Primeiros 3 pontos: {objp[:3]}")
        return objp

    def load_images(self, left_images_dir, right_images_dir):
        """Carrega listas de imagens ordenadas"""
        left_patterns = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        right_patterns = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        
        images_left = []
        images_right = []
        
        for pattern in left_patterns:
            found_left = sorted(glob.glob(os.path.join(left_images_dir, pattern)))
            images_left.extend(found_left)
            if found_left:
                print(f"[DEBUG] Encontradas {len(found_left)} imagens esquerdas com padrão {pattern}")
        
        for pattern in right_patterns:
            found_right = sorted(glob.glob(os.path.join(right_images_dir, pattern)))
            images_right.extend(found_right)
            if found_right:
                print(f"[DEBUG] Encontradas {len(found_right)} imagens direitas com padrão {pattern}")
        
        print(f"[DEBUG] Total de imagens carregadas: {len(images_left)} esquerdas, {len(images_right)} direitas")
        
        if len(images_left) != len(images_right):
            raise RuntimeError(f"Número diferente de imagens: {len(images_left)} esquerda vs {len(images_right)} direita")
        
        if len(images_left) == 0:
            raise RuntimeError("Nenhuma imagem encontrada nos diretórios especificados")
        
        # Debug: mostra os primeiros arquivos
        print(f"[DEBUG] Primeiras 3 imagens esquerdas: {[os.path.basename(img) for img in images_left[:3]]}")
        print(f"[DEBUG] Primeiras 3 imagens direitas: {[os.path.basename(img) for img in images_right[:3]]}")
        
        return images_left, images_right

    def find_corners(self, images_left, images_right, max_images=None):
        """Encontra cantos do tabuleiro - versão simplificada"""
        objp = self._prepare_object_points()
        
        # Critérios para refinamento subpixel
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
        print(f"[DEBUG] Critérios de refinamento: {criteria}")
        
        # Flags simplificadas para detecção de cantos
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        print(f"[DEBUG] Flags de detecção: {flags}")
        
        valid_pairs = 0
        total_pairs = len(images_left)
        
        if max_images:
            total_pairs = min(total_pairs, max_images)
        
        print(f"[INFO] Processando {total_pairs} pares de imagens...")
        
        for i, (img_left_path, img_right_path) in enumerate(zip(images_left[:total_pairs], images_right[:total_pairs])):
            print(f"\n[DEBUG] === Processando par {i+1}/{total_pairs} ===")
            print(f"[DEBUG] Esquerda: {os.path.basename(img_left_path)}")
            print(f"[DEBUG] Direita: {os.path.basename(img_right_path)}")
            
            # Carrega imagens
            img_left = cv2.imread(img_left_path)
            img_right = cv2.imread(img_right_path)
            
            if img_left is None:
                print(f"[ERROR] Não foi possível carregar imagem esquerda: {img_left_path}")
                continue
            if img_right is None:
                print(f"[ERROR] Não foi possível carregar imagem direita: {img_right_path}")
                continue
            
            print(f"[DEBUG] Dimensões esquerda: {img_left.shape}")
            print(f"[DEBUG] Dimensões direita: {img_right.shape}")
            
            # Converte para escala de cinza
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
            
            print(f"[DEBUG] Dimensões cinza esquerda: {gray_left.shape}")
            print(f"[DEBUG] Dimensões cinza direita: {gray_right.shape}")
            
            # Salva dimensões da imagem (uma vez)
            if self.image_size is None:
                self.image_size = (gray_left.shape[1], gray_left.shape[0])  # (width, height)
                print(f"[DEBUG] Tamanho da imagem definido: {self.image_size}")
            
            # Encontra cantos
            print(f"[DEBUG] Procurando cantos para tabuleiro {self.chessboard_size}...")
            retL, corners_left = cv2.findChessboardCorners(gray_left, self.chessboard_size, flags=flags)
            retR, corners_right = cv2.findChessboardCorners(gray_right, self.chessboard_size, flags=flags)
            
            print(f"[DEBUG] Cantos encontrados - Esquerda: {retL}, Direita: {retR}")
            
            if retL:
                print(f"[DEBUG] Quantidade de cantos esquerda: {len(corners_left)}")
                print(f"[DEBUG] Primeiro canto esquerda: {corners_left[0].flatten()}")
            
            if retR:
                print(f"[DEBUG] Quantidade de cantos direita: {len(corners_right)}")
                print(f"[DEBUG] Primeiro canto direita: {corners_right[0].flatten()}")
            
            if not (retL and retR):
                print(f"[WARNING] Cantos não encontrados no par {i+1} - Esquerda: {retL}, Direita: {retR}")
                continue
            
            # Refinamento subpixel
            print(f"[DEBUG] Aplicando refinamento subpixel...")
            corners_left_refined = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
            corners_right_refined = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
            
            # Verifica se o refinamento mudou os pontos
            diff_left = np.mean(np.abs(corners_left - corners_left_refined))
            diff_right = np.mean(np.abs(corners_right - corners_right_refined))
            print(f"[DEBUG] Diferença média após refinamento - Esquerda: {diff_left:.4f}, Direita: {diff_right:.4f}")
            
            # Adiciona aos datasets
            self.objpoints.append(objp)
            self.imgpoints_left.append(corners_left_refined)
            self.imgpoints_right.append(corners_right_refined)
            
            valid_pairs += 1
            print(f"[INFO] Par {i+1} aceito - Total válidos: {valid_pairs}")
        
        print(f"\n[INFO] === RESUMO DETECÇÃO DE CANTOS ===")
        print(f"[INFO] {valid_pairs} pares válidos de {total_pairs} processados")
        print(f"[INFO] Taxa de sucesso: {valid_pairs/total_pairs*100:.1f}%")
        
        if valid_pairs < 10:
            print("[WARNING] Poucos pares válidos para calibração confiável!")
        
        return valid_pairs

    def calibrate_individual(self):
        """Calibração individual das câmeras"""
        if len(self.objpoints) == 0:
            raise RuntimeError("Nenhum ponto detectado para calibração")
        
        print(f"\n[DEBUG] === CALIBRAÇÃO INDIVIDUAL ===")
        print(f"[DEBUG] Número de conjuntos de pontos: {len(self.objpoints)}")
        print(f"[DEBUG] Tamanho da imagem: {self.image_size}")
        
        # Calibração câmera esquerda
        print(f"[DEBUG] Iniciando calibração da câmera esquerda...")
        retL, self.mtx_left, self.dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_left, self.image_size, None, None)
        
        print(f"[DEBUG] Matriz intrínseca esquerda:\n{self.mtx_left}")
        print(f"[DEBUG] Coeficientes de distorção esquerda: {self.dist_left.flatten()}")
        print(f"[DEBUG] Número de vetores de rotação esquerda: {len(rvecs_left)}")
        print(f"[DEBUG] Número de vetores de translação esquerda: {len(tvecs_left)}")
        
        # Calibração câmera direita
        print(f"[DEBUG] Iniciando calibração da câmera direita...")
        retR, self.mtx_right, self.dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_right, self.image_size, None, None)
        
        print(f"[DEBUG] Matriz intrínseca direita:\n{self.mtx_right}")
        print(f"[DEBUG] Coeficientes de distorção direita: {self.dist_right.flatten()}")
        print(f"[DEBUG] Número de vetores de rotação direita: {len(rvecs_right)}")
        print(f"[DEBUG] Número de vetores de translação direita: {len(tvecs_right)}")
        
        print(f"[INFO] Erro RMS câmera esquerda: {retL:.4f}")
        print(f"[INFO] Erro RMS câmera direita: {retR:.4f}")
        
        # Análise dos parâmetros
        fx_left, fy_left = self.mtx_left[0,0], self.mtx_left[1,1]
        cx_left, cy_left = self.mtx_left[0,2], self.mtx_left[1,2]
        fx_right, fy_right = self.mtx_right[0,0], self.mtx_right[1,1]
        cx_right, cy_right = self.mtx_right[0,2], self.mtx_right[1,2]
        
        print(f"[DEBUG] Distância focal esquerda: fx={fx_left:.1f}, fy={fy_left:.1f}")
        print(f"[DEBUG] Centro principal esquerda: cx={cx_left:.1f}, cy={cy_left:.1f}")
        print(f"[DEBUG] Distância focal direita: fx={fx_right:.1f}, fy={fy_right:.1f}")
        print(f"[DEBUG] Centro principal direita: cx={cx_right:.1f}, cy={cy_right:.1f}")
        
        # Validação básica das matrizes
        if retL > 1.0 or retR > 1.0:
            print("[WARNING] Erro RMS alto nas calibrações individuais!")
        
        if abs(fx_left - fy_left) > 50 or abs(fx_right - fy_right) > 50:
            print("[WARNING] Grande diferença entre fx e fy - possível problema de aspecto!")
        
        return retL, retR

    def stereo_calibrate(self):
        """Calibração estéreo das câmeras"""
        if self.mtx_left is None or self.mtx_right is None:
            raise RuntimeError("Calibração individual deve ser executada primeiro")
        
        print(f"\n[DEBUG] === CALIBRAÇÃO ESTÉREO ===")
        print(f"[DEBUG] Iniciando calibração estéreo com {len(self.objpoints)} conjuntos de pontos...")
        
        # Critérios de otimização
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-6)
        print(f"[DEBUG] Critérios de otimização: {criteria}")
        
        # Flags para calibração estéreo
        flags = 0
        print(f"[DEBUG] Flags de calibração estéreo: {flags}")
        
        # Calibração estéreo
        print(f"[DEBUG] Executando cv2.stereoCalibrate...")
        ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints,
            self.imgpoints_left,
            self.imgpoints_right,
            self.mtx_left,
            self.dist_left,
            self.mtx_right,
            self.dist_right,
            self.image_size,
            criteria=criteria,
            flags=flags
        )
        
        # Atualiza parâmetros
        self.mtx_left = mtx_left
        self.dist_left = dist_left
        self.mtx_right = mtx_right
        self.dist_right = dist_right
        self.R = R
        self.T = T
        self.E = E
        self.F = F
        
        print(f"[DEBUG] Matriz de rotação R:\n{R}")
        print(f"[DEBUG] Vetor de translação T: {T.flatten()}")
        print(f"[DEBUG] Matriz essencial E:\n{E}")
        print(f"[DEBUG] Matriz fundamental F:\n{F}")
        
        print(f"[INFO] Erro RMS calibração estéreo: {ret:.4f}")
        
        # Validações detalhadas
        baseline = np.linalg.norm(T)
        print(f"[INFO] Baseline (distância entre câmeras): {baseline*1000:.1f} mm")
        
        # Análise da rotação
        rotation_angle = np.arccos((np.trace(R) - 1) / 2) * 180 / np.pi
        print(f"[DEBUG] Ângulo de rotação entre câmeras: {rotation_angle:.2f} graus")
        
        # Análise da translação
        tx, ty, tz = T.flatten()
        print(f"[DEBUG] Translação - X: {tx*1000:.1f}mm, Y: {ty*1000:.1f}mm, Z: {tz*1000:.1f}mm")
        
        if baseline < 0.05:  # Menos que 5cm
            print("[WARNING] Baseline muito pequena para boa precisão de profundidade")
        
        if baseline > 1.0:  # Mais que 1m
            print("[WARNING] Baseline muito grande - pode causar problemas de correspondência")
        
        if abs(ty) > 0.02:  # Desalinhamento vertical > 2cm
            print("[WARNING] Grande desalinhamento vertical entre as câmeras")
        
        if ret > 1.0:
            print("[WARNING] Erro RMS alto na calibração estéreo!")
        
        return ret

    def rectify(self, alpha=0.5):
        """Retificação estéreo"""
        if self.R is None or self.T is None:
            raise RuntimeError("Calibração estéreo deve ser executada primeiro")
        
        print(f"\n[DEBUG] === RETIFICAÇÃO ESTÉREO ===")
        print(f"[DEBUG] Iniciando retificação com alpha={alpha}")
        
        # Retificação estéreo
        R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
            self.mtx_left, self.dist_left,
            self.mtx_right, self.dist_right,
            self.image_size, self.R, self.T,
            alpha=alpha,
            flags=cv2.CALIB_ZERO_DISPARITY
        )
        
        print(f"[DEBUG] Matriz de rotação R1 (esquerda):\n{R1}")
        print(f"[DEBUG] Matriz de rotação R2 (direita):\n{R2}")
        print(f"[DEBUG] Matriz de projeção P1 (esquerda):\n{P1}")
        print(f"[DEBUG] Matriz de projeção P2 (direita):\n{P2}")
        print(f"[DEBUG] Matriz Q de reprojeção:\n{Q}")
        
        # Salva parâmetros
        self.P1 = P1
        self.P2 = P2
        self.Q = Q
        self.roi_left = roi_left
        self.roi_right = roi_right
        
        print(f"[DEBUG] Gerando mapas de retificação...")
        
        # Gera mapas de retificação
        self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(
            self.mtx_left, self.dist_left, R1, P1, self.image_size, cv2.CV_16SC2)
        
        self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(
            self.mtx_right, self.dist_right, R2, P2, self.image_size, cv2.CV_16SC2)
        
        print(f"[DEBUG] Mapas de retificação gerados com sucesso")
        print(f"[DEBUG] Tipo dos mapas: {type(self.left_map1)}, {type(self.left_map2)}")
        
        print(f"[INFO] ROI esquerda: {roi_left}")
        print(f"[INFO] ROI direita: {roi_right}")
        
        # Análise das ROIs
        roi_left_area = roi_left[2] * roi_left[3] if roi_left[2] > 0 and roi_left[3] > 0 else 0
        roi_right_area = roi_right[2] * roi_right[3] if roi_right[2] > 0 and roi_right[3] > 0 else 0
        total_area = self.image_size[0] * self.image_size[1]
        
        print(f"[DEBUG] Área ROI esquerda: {roi_left_area} ({roi_left_area/total_area*100:.1f}% da imagem)")
        print(f"[DEBUG] Área ROI direita: {roi_right_area} ({roi_right_area/total_area*100:.1f}% da imagem)")

    def validate_calibration(self):
        """Validação da calibração através do erro epipolar"""
        if self.F is None:
            raise RuntimeError("Matriz fundamental não calculada")
        
        print(f"\n[DEBUG] === VALIDAÇÃO DE CALIBRAÇÃO ===")
        print(f"[DEBUG] Calculando erro epipolar para {len(self.imgpoints_left)} pares...")
        
        total_error = 0
        total_points = 0
        errors_per_image = []
        
        for idx, (pts_left, pts_right) in enumerate(zip(self.imgpoints_left, self.imgpoints_right)):
            # Converte formato dos pontos
            pts_left = pts_left.reshape(-1, 2)
            pts_right = pts_right.reshape(-1, 2)
            
            # Calcula linhas epipolares
            lines_right = cv2.computeCorrespondEpilines(pts_left.reshape(-1, 1, 2), 1, self.F)
            lines_left = cv2.computeCorrespondEpilines(pts_right.reshape(-1, 1, 2), 2, self.F)
            
            lines_right = lines_right.reshape(-1, 3)
            lines_left = lines_left.reshape(-1, 3)
            
            # Calcula erro epipolar para esta imagem
            image_error = 0
            for pt_l, pt_r, line_r, line_l in zip(pts_left, pts_right, lines_right, lines_left):
                # Distância ponto-linha
                d_right = abs(line_r[0]*pt_r[0] + line_r[1]*pt_r[1] + line_r[2]) / np.sqrt(line_r[0]**2 + line_r[1]**2)
                d_left = abs(line_l[0]*pt_l[0] + line_l[1]*pt_l[1] + line_l[2]) / np.sqrt(line_l[0]**2 + line_l[1]**2)
                image_error += d_left + d_right
                total_error += d_left + d_right
                total_points += 2
            
            avg_error_this_image = image_error / (len(pts_left) * 2)
            errors_per_image.append(avg_error_this_image)
            
            if idx < 5:  # Mostra detalhes das primeiras 5 imagens
                print(f"[DEBUG] Imagem {idx+1}: erro médio = {avg_error_this_image:.4f} pixels")
        
        mean_error = total_error / total_points
        std_error = np.std(errors_per_image)
        min_error = np.min(errors_per_image)
        max_error = np.max(errors_per_image)
        
        print(f"[DEBUG] Estatísticas do erro epipolar:")
        print(f"[DEBUG] - Erro médio: {mean_error:.4f} pixels")
        print(f"[DEBUG] - Desvio padrão: {std_error:.4f} pixels")
        print(f"[DEBUG] - Erro mínimo: {min_error:.4f} pixels")
        print(f"[DEBUG] - Erro máximo: {max_error:.4f} pixels")
        
        print(f"[INFO] Erro médio epipolar: {mean_error:.4f} pixels")
        
        if mean_error > 1.0:
            print("[WARNING] Erro epipolar alto! Calibração pode não estar boa.")
        elif mean_error < 0.5:
            print("[INFO] Excelente calibração!")
        else:
            print("[INFO] Calibração aceitável.")
        
        return mean_error

    def run_full_calibration(self, left_dir, right_dir, max_images=None):
        """Pipeline completo de calibração"""
        print(f"\n[DEBUG] === PIPELINE COMPLETO DE CALIBRAÇÃO ===")
        print(f"[DEBUG] Diretório esquerdo: {left_dir}")
        print(f"[DEBUG] Diretório direito: {right_dir}")
        print(f"[DEBUG] Máximo de imagens: {max_images}")
        print(f"[DEBUG] Tamanho do tabuleiro: {self.chessboard_size}")
        print(f"[DEBUG] Tamanho do quadrado: {self.square_size}m")
        
        # Carrega imagens
        images_left, images_right = self.load_images(left_dir, right_dir)
        
        # Encontra cantos
        valid_pairs = self.find_corners(images_left, images_right, max_images)
        
        if valid_pairs == 0:
            raise RuntimeError("Nenhum par válido encontrado!")
        
        # Calibração individual
        self.calibrate_individual()
        
        # Calibração estéreo
        self.stereo_calibrate()
        
        # Retificação
        self.rectify()
        
        # Validação
        self.validate_calibration()
        
        print(f"\n[INFO] === CALIBRAÇÃO COMPLETA COM SUCESSO! ===")
        return True

    def save_calibration(self, path):
        """Salva parâmetros de calibração"""
        print(f"[DEBUG] Salvando calibração em: {path}")
        
        calibration_data = {
            'mtx_left': self.mtx_left,
            'dist_left': self.dist_left,
            'mtx_right': self.mtx_right,
            'dist_right': self.dist_right,
            'R': self.R,
            'T': self.T,
            'E': self.E,
            'F': self.F,
            'P1': self.P1,
            'P2': self.P2,
            'Q': self.Q,
            'left_map1': self.left_map1,
            'left_map2': self.left_map2,
            'right_map1': self.right_map1,
            'right_map2': self.right_map2,
            'roi_left': self.roi_left,
            'roi_right': self.roi_right,
            'image_size': self.image_size,
            'chessboard_size': self.chessboard_size,
            'square_size': self.square_size
        }
        
        # Debug: mostra tamanhos dos dados
        for key, value in calibration_data.items():
            if hasattr(value, 'shape'):
                print(f"[DEBUG] {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"[DEBUG] {key}: {type(value)}")
        
        with open(path, 'wb') as f:
            pickle.dump(calibration_data, f)
        
        print(f"[INFO] Calibração salva em: {path}")

    def load_calibration(self, path):
        """Carrega parâmetros de calibração"""
        print(f"[DEBUG] Carregando calibração de: {path}")
        
        with open(path, 'rb') as f:
            calibration_data = pickle.load(f)
        
        print(f"[DEBUG] Dados carregados: {list(calibration_data.keys())}")
        
        for key, value in calibration_data.items():
            setattr(self, key, value)
            if hasattr(value, 'shape'):
                print(f"[DEBUG] {key}: shape={value.shape}, dtype={value.dtype}")
        
        print(f"[INFO] Calibração carregada de: {path}")

    def rectify_images(self, img_left, img_right):
        """Retifica par de imagens"""
        if self.left_map1 is None or self.right_map1 is None:
            raise RuntimeError("Mapas de retificação não foram gerados")
        
        print(f"[DEBUG] Retificando imagens com dimensões: {img_left.shape}, {img_right.shape}")
        
        rect_left = cv2.remap(img_left, self.left_map1, self.left_map2, cv2.INTER_LINEAR)
        rect_right = cv2.remap(img_right, self.right_map1, self.right_map2, cv2.INTER_LINEAR)
        
        print(f"[DEBUG] Imagens retificadas com dimensões: {rect_left.shape}, {rect_right.shape}")
        
        return rect_left, rect_right
    

    def test_rectification(self, img_left, img_right, num_lines=10):
        """Testa retificação desenhando linhas epipolares em imagens individuais"""
        if self.left_map1 is None or self.right_map1 is None:
            raise RuntimeError("Mapas de retificação não foram gerados")
        
        print(f"[DEBUG] Testando retificação com {num_lines} linhas epipolares")
        
        # Retifica as imagens
        rect_left, rect_right = self.rectify_images(img_left, img_right)
        
        # Cria cópias para desenhar
        display_left = rect_left.copy()
        display_right = rect_right.copy()
        
        # Desenha linhas epipolares horizontais
        height, width = rect_left.shape[:2]
        
        for i in range(num_lines):
            y = int(height * (i + 1) / (num_lines + 1))
            
            # Linha na imagem esquerda
            cv2.line(display_left, (0, y), (width, y), (0, 255, 0), 2)
            
            # Linha na imagem direita
            cv2.line(display_right, (0, y), (width, y), (0, 255, 0), 2)
        
        # Redimensiona para visualização se necessário
        max_height = 800
        if height > max_height:
            scale = max_height / height
            new_width = int(width * scale)
            display_left = cv2.resize(display_left, (new_width, max_height))
            display_right = cv2.resize(display_right, (new_width, max_height))
        
        # Mostra as imagens lado a lado
        cv2.imshow('Rectified Left', display_left)
        cv2.imshow('Rectified Right', display_right)
        
        print(f"[INFO] Imagens retificadas exibidas. Pressione qualquer tecla para continuar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return display_left, display_right
    