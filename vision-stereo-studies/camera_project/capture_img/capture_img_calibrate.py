import os
import time
import cv2
from pathlib import Path
import sys

# Adiciona o diretório raiz do projeto
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from camera_project.stereo_vision.capture import StereoCameras

# Caminhos onde as imagens serão salvas
LEFT_DIR = "calib_data/left"
RIGHT_DIR = "calib_data/right"

# Cria diretórios se não existirem
os.makedirs(LEFT_DIR, exist_ok=True)
os.makedirs(RIGHT_DIR, exist_ok=True)

# Inicializa câmeras
cameras = StereoCameras(left_cam_id=1, right_cam_id=2)

print("[INFO] Iniciando captura automática de 50 imagens (25 pares).")
print("[INFO] Intervalo de 10 segundos entre cada captura.")
print("[INFO] Pressione ESC a qualquer momento para cancelar.")

image_counter = 1
total_images = 25  # 25 pares (50 imagens no total)

while image_counter <= total_images:
    frame_left, frame_right = cameras.get_frames()

    # Mostra as imagens em tempo real
    frame_left_resized = cv2.resize(frame_left, (640, 480))
    frame_right_resized = cv2.resize(frame_right, (640, 480))
    combined = cv2.hconcat([frame_left_resized, frame_right_resized])
    cv2.imshow("Captura Calibração (ESQ + DIR)", combined)

    # Contagem regressiva antes de capturar
    for i in range(10, 0, -1):
        print(f"[INFO] Próxima captura em: {i}s (Pressione ESC para cancelar)", end="\r")
        key = cv2.waitKey(1000)  # Aguarda 1s e verifica se ESC foi pressionado
        if key == 27:  # ESC
            cameras.release()
            cv2.destroyAllWindows()
            exit()

    # Salva imagens com numeração
    img_name = f"img_{image_counter:02d}.png"
    path_left = os.path.join(LEFT_DIR, img_name)
    path_right = os.path.join(RIGHT_DIR, img_name)

    cv2.imwrite(path_left, frame_left)
    cv2.imwrite(path_right, frame_right)

    print(f"\n[✓] Par {image_counter}/{total_images} salvo: {path_left} e {path_right}")
    image_counter += 1

# Libera recursos
cameras.release()
cv2.destroyAllWindows()
print("[INFO] Captura concluída! Todas as imagens foram salvas.")