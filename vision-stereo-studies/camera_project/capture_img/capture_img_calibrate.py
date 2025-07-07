import os
import time
import cv2
import numpy as np
from pathlib import Path
import sys
import csv

# Adiciona o diretório raiz do projeto
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from stereo_vision.capture import StereoCameras
LEFT_DIR = r"C:\Venv\Robot_project\robot-project\vision-stereo-studies\camera_project\datasets\images\seq02\left"
RIGHT_DIR = r"C:\Venv\Robot_project\robot-project\vision-stereo-studies\camera_project\datasets\images\seq02\right"

# === CONFIGURAÇÕES MELHORADAS ===
CHESSBOARD_SIZE = (8, 5)
CHESSBOARD_SIZES = [(8, 5), (9, 6), (7, 4)]  # Tamanhos alternativos
TOTAL_IMAGES = 100
SQUARE_CAPTURE_THRESHOLD = 2000  # Reduzido para aceitar mais imagens
CAPTURE_DELAY = 3
LEFT_CAM_ID = 0
RIGHT_CAM_ID = 2

# === PREPARAÇÃO ===
os.makedirs(LEFT_DIR, exist_ok=True)
os.makedirs(RIGHT_DIR, exist_ok=True)

# === FUNÇÕES AUXILIARES MELHORADAS ===
def configurar_cameras(cameras):
    """
    Configura as câmeras para máxima qualidade
    """
    try:
        # Configurações para webcam 1080p
        cameras.left_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cameras.left_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cameras.right_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cameras.right_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Configurações de qualidade
        cameras.left_cap.set(cv2.CAP_PROP_FPS, 30)
        cameras.right_cap.set(cv2.CAP_PROP_FPS, 30)
        cameras.left_cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
        cameras.right_cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
        
        # Foco automático (se suportado)
        cameras.left_cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        cameras.right_cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        # Verificar se as configurações foram aplicadas
        width_l = cameras.left_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height_l = cameras.left_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width_r = cameras.right_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height_r = cameras.right_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        print(f"[INFO] Câmera esquerda: {width_l}x{height_l}")
        print(f"[INFO] Câmera direita: {width_r}x{height_r}")
        
    except Exception as e:
        print(f"[AVISO] Erro ao configurar câmeras: {e}")

def preprocessar_imagem(gray):
    """
    Pré-processa a imagem para melhorar a detecção
    """
    # Equalização de histograma adaptativa
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Filtro gaussiano para reduzir ruído
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    return blurred

def detectar_tabuleiro_multiplo(gray):
    """
    Tenta detectar tabuleiro com múltiplos tamanhos
    """
    # Flags melhoradas para detecção
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS
    
    for size in CHESSBOARD_SIZES:
        ret, corners = cv2.findChessboardCorners(gray, size, flags=flags)
        if ret:
            # Refinamento subpixel
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
            )
            return ret, corners, size
    
    return False, None, CHESSBOARD_SIZE

def calcular_qualidade_melhorada(corners, image_shape, board_size):
    """
    Calcula a qualidade da detecção com escala melhorada
    """
    if corners is None or len(corners) < 4:
        return 0.0
    
    corners_2d = corners.reshape(-1, 2)
    
    # 1. Área do tabuleiro (hull convexo)
    hull = cv2.convexHull(corners_2d.astype(np.float32))
    area = cv2.contourArea(hull)
    max_area = image_shape[0] * image_shape[1]
    coverage_score = (area / max_area) * 8000  # Peso maior para área
    
    # 2. Distribuição dos pontos (dispersão)
    center_x, center_y = image_shape[1]//2, image_shape[0]//2
    distances = np.sqrt((corners_2d[:, 0] - center_x)**2 + (corners_2d[:, 1] - center_y)**2)
    distribution_score = np.std(distances) * 3  # Melhor distribuição
    
    # 3. Penalidade por proximidade das bordas
    border_margin = 80
    min_x, max_x = np.min(corners_2d[:, 0]), np.max(corners_2d[:, 0])
    min_y, max_y = np.min(corners_2d[:, 1]), np.max(corners_2d[:, 1])
    
    border_penalty = 0
    if min_x < border_margin or max_x > image_shape[1] - border_margin:
        border_penalty += 800
    if min_y < border_margin or max_y > image_shape[0] - border_margin:
        border_penalty += 800
    
    # 4. Bônus pelo tamanho do tabuleiro
    expected_points = board_size[0] * board_size[1]
    if len(corners_2d) == expected_points:
        size_bonus = 500
    else:
        size_bonus = 0
    
    final_score = coverage_score + distribution_score - border_penalty + size_bonus
    return max(0, final_score)

def draw_quality_bar(img, score, max_score=8000):
    """
    Desenha uma barra de qualidade na imagem
    """
    bar_width = 300
    bar_height = 25
    x, y = 20, img.shape[0] - 60
    pct = min(score / max_score, 1.0)
    filled = int(bar_width * pct)

    # Define cor de acordo com qualidade
    if score < 1000:
        color = (0, 0, 255)  # Vermelho
        status = "Ruim"
    elif score < 2000:
        color = (0, 165, 255)  # Laranja
        status = "Razoável"
    elif score < 3500:
        color = (0, 255, 255)  # Amarelo
        status = "Bom"
    else:
        color = (0, 255, 0)  # Verde
        status = "Excelente"

    # Desenha a barra
    cv2.rectangle(img, (x, y), (x + bar_width, y + bar_height), (50, 50, 50), -1)
    cv2.rectangle(img, (x, y), (x + filled, y + bar_height), color, -1)
    cv2.rectangle(img, (x, y), (x + bar_width, y + bar_height), (255, 255, 255), 2)
    cv2.putText(img, f"Qualidade: {score:.0f} ({status})", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def draw_quality_info(img, corners, score, board_size):
    """
    Desenha informações detalhadas sobre a qualidade
    """
    if corners is not None:
        corners_2d = corners.reshape(-1, 2)
        hull = cv2.convexHull(corners_2d.astype(np.float32))
        area = cv2.contourArea(hull)
        
        # Informações detalhadas
        info_text = [
            f"Pontos: {len(corners_2d)}/{board_size[0]*board_size[1]}",
            f"Area: {area:.0f}px²",
            f"Cobertura: {(area/(img.shape[0]*img.shape[1])*100):.1f}%",
            f"Tabuleiro: {board_size[0]}x{board_size[1]}"
        ]
        
        # Fundo para o texto
        for i, text in enumerate(info_text):
            y_pos = 70 + i * 25
            cv2.rectangle(img, (10, y_pos - 18), (250, y_pos + 7), (0, 0, 0), -1)
            cv2.putText(img, text, (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# === INICIALIZAÇÃO ===
cameras = StereoCameras(left_cam_id=LEFT_CAM_ID, right_cam_id=RIGHT_CAM_ID)
configurar_cameras(cameras)

image_counter = 1
log_entries = []

print(f"[INFO] Captura automática de {TOTAL_IMAGES} pares")
print(f"[INFO] Threshold de qualidade: {SQUARE_CAPTURE_THRESHOLD}")
print("[INFO] Pressione ESC para cancelar")

# === LOOP PRINCIPAL ===
try:
    while image_counter <= TOTAL_IMAGES:
        frame_left, frame_right = cameras.get_frames()

        if frame_left is None or frame_right is None:
            print("[ERRO] Falha na captura das câmeras")
            break

        # Pré-processamento melhorado
        gray_left = preprocessar_imagem(cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY))
        gray_right = preprocessar_imagem(cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY))

        # Detecção com múltiplos tamanhos
        retL, corners_left, size_left = detectar_tabuleiro_multiplo(gray_left)
        retR, corners_right, size_right = detectar_tabuleiro_multiplo(gray_right)

        # Garantir que ambos usam o mesmo tamanho
        if retL and retR and size_left != size_right:
            # Usar o tamanho padrão se houver conflito
            retL, corners_left = cv2.findChessboardCorners(gray_left, CHESSBOARD_SIZE)
            retR, corners_right = cv2.findChessboardCorners(gray_right, CHESSBOARD_SIZE)
            size_left = size_right = CHESSBOARD_SIZE

        vis_left = frame_left.copy()
        vis_right = frame_right.copy()
        
        # Calcular qualidade melhorada
        quality_left = calcular_qualidade_melhorada(corners_left, gray_left.shape, size_left) if retL else 0.0
        quality_right = calcular_qualidade_melhorada(corners_right, gray_right.shape, size_right) if retR else 0.0
        quality_score = min(quality_left, quality_right)

        # Desenhar os cantos detectados
        if retL:
            cv2.drawChessboardCorners(vis_left, size_left, corners_left, retL)
        if retR:
            cv2.drawChessboardCorners(vis_right, size_right, corners_right, retR)

        # Redimensionar para visualização (mantém proporção)
        h, w = vis_left.shape[:2]
        if w > 640:
            scale = 640 / w
            new_w, new_h = int(w * scale), int(h * scale)
            left_resized = cv2.resize(vis_left, (new_w, new_h))
            right_resized = cv2.resize(vis_right, (new_w, new_h))
        else:
            left_resized = vis_left
            right_resized = vis_right

        # Combinar imagens
        combined = cv2.hconcat([left_resized, right_resized])
        
        # Adicionar informações visuais
        draw_quality_bar(combined, quality_score)
        if retL and retR:
            draw_quality_info(combined, corners_left, quality_score, size_left)

        # Status da captura
        status = f"[{image_counter}/{TOTAL_IMAGES}] "

        if retL and retR:
            if quality_score > SQUARE_CAPTURE_THRESHOLD:
                status += f"✅ Capturando ({quality_score:.0f}) em {CAPTURE_DELAY}s..."
                cv2.putText(combined, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Calibração Estéreo - Melhorada", combined)
                
                # Aguarda o tempo de captura
                key = cv2.waitKey(CAPTURE_DELAY * 1000)
                
                if key != 27:  # Se não pressionou ESC durante a espera
                    # Salva as imagens originais (não redimensionadas)
                    img_name = f"img_{image_counter:02d}.png"
                    cv2.imwrite(os.path.join(LEFT_DIR, img_name), frame_left)
                    cv2.imwrite(os.path.join(RIGHT_DIR, img_name), frame_right)
                    
                    print(f"[✓] Par {image_counter} salvo: {img_name} (qualidade {quality_score:.0f}, {size_left[0]}x{size_left[1]})")
                    log_entries.append((img_name, quality_score, f"{size_left[0]}x{size_left[1]}", 
                                      f"{quality_left:.0f}", f"{quality_right:.0f}"))
                    image_counter += 1
                else:
                    break
            else:
                status += f"⚠️ Qualidade baixa ({quality_score:.0f}) - mínimo {SQUARE_CAPTURE_THRESHOLD}"
                cv2.putText(combined, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                key = cv2.waitKey(100)
        else:
            status += "❌ Tabuleiro não detectado em ambas as câmeras"
            cv2.putText(combined, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            key = cv2.waitKey(100)

        cv2.imshow("Calibração Estéreo - Melhorada", combined)

        if key == 27:  # ESC
            print("[INFO] Cancelado pelo usuário.")
            break

except KeyboardInterrupt:
    print("[INFO] Interrompido pelo usuário (Ctrl+C)")
except Exception as e:
    print(f"[ERRO] Ocorreu um erro durante a captura: {e}")
    import traceback
    traceback.print_exc()

finally:
    # === FINALIZAÇÃO ===
    cameras.release()
    cv2.destroyAllWindows()

    # Salvar log detalhado
    log_path = 'seq02.csv'
    with open(log_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Imagem", "Qualidade_Total", "Tamanho_Tabuleiro", "Qualidade_Esquerda", "Qualidade_Direita"])
        writer.writerows(log_entries)

    print(f"\n[INFO] === RESUMO DA CAPTURA ===")
    print(f"[INFO] Imagens capturadas: {image_counter - 1}/{TOTAL_IMAGES}")
    print(f"[INFO] Log salvo em: {log_path}")
    
    if log_entries:
        avg_quality = sum(float(entry[1]) for entry in log_entries) / len(log_entries)
        max_quality = max(float(entry[1]) for entry in log_entries)
        print(f"[INFO] Qualidade média: {avg_quality:.0f}")
        print(f"[INFO] Qualidade máxima: {max_quality:.0f}")
    
    print("[INFO] Captura finalizada!")