import cv2
from pathlib import Path
import sys

# Adiciona o diretório raiz do projeto
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from camera_project.stereo_vision.capture import StereoCameras

def main():
    print("[INFO] Iniciando teste das câmeras estéreo...")

    # Inicializa as câmeras com os IDs (ajuste se necessário)
    cameras = StereoCameras(left_cam_id=1, right_cam_id=2)

    print("[INFO] Pressione ESC para encerrar.")
    while True:
        # Captura os dois frames
        frame_left, frame_right = cameras.get_frames()

        # Redimensiona para exibir lado a lado (opcional)
        frame_left_resized = cv2.resize(frame_left, (640, 480))
        frame_right_resized = cv2.resize(frame_right, (640, 480))

        # Combina as duas imagens horizontalmente
        combined = cv2.hconcat([frame_left_resized, frame_right_resized])

        # Mostra na tela
        cv2.imshow("Câmeras lado a lado", combined)

        # Sai com ESC
        key = cv2.waitKey(1)
        if key == 27:
            break

    cameras.release()
    cv2.destroyAllWindows()
    print("[✓] Encerrado com sucesso.")

if __name__ == "__main__":
    main()
