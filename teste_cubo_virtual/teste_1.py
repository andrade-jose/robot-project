import cv2
import numpy as np
import time

# --- A. PARÂMETROS E FORMAS ---
# Calibração da Câmera
CAMERA_MATRIX = np.array([
    [800, 0, 320],    # fx, 0, cx
    [0, 800, 240],    # 0, fy, cy
    [0, 0, 1]         # 0, 0, 1
], dtype=np.float64)
DIST_COEFFS = np.zeros((5, 1), dtype=np.float64)

# Parâmetros do Marcador
MARKER_SIZE_IN_METERS = 0.05
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
ARUCO_PARAMETERS = cv2.aruco.DetectorParameters()

# Vertices e arestas do cubo 3D
VERTICES_3D = np.float32([
    [-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0],
    [-0.5, -0.5, 1], [0.5, -0.5, 1], [0.5, 0.5, 1], [-0.5, 0.5, 1]
]) * MARKER_SIZE_IN_METERS

ARESTAS = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]

def draw_3d_object_on_opencv(frame, rvec, tvec):
    """
    Desenha um objeto 3D projetado na imagem 2D do OpenCV.
    """
    # Projeta os pontos 3D dos vértices para o plano 2D da imagem
    img_points, _ = cv2.projectPoints(
        VERTICES_3D, rvec, tvec, CAMERA_MATRIX, DIST_COEFFS
    )
    img_points = np.int32(img_points).reshape(-1, 2)

    # Desenha as arestas do cubo
    for i, j in ARESTAS:
        cv2.line(frame, tuple(img_points[i]), tuple(img_points[j]), (255, 0, 0), 2)
    
    # Desenha os pontos dos vértices
    for point in img_points:
        cv2.circle(frame, tuple(point), 3, (0, 0, 255), -1)

# --- LOOP PRINCIPAL ---
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Erro: Não foi possível abrir a câmera.")
        return

    print("✅ Janela do OpenCV pronta. Posicione o marcador com ID 0.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)

        # Detecta os marcadores
        corners, ids, _ = cv2.aruco.detectMarkers(
            frame, ARUCO_DICT, parameters=ARUCO_PARAMETERS
        )

        if ids is not None:
            # Filtra e processa apenas o marcador com ID 0
            for i, detected_id in enumerate(ids):
                if detected_id[0] == 0:
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners[i:i+1], MARKER_SIZE_IN_METERS, CAMERA_MATRIX, DIST_COEFFS
                    )
                    
                    # Desenha o cubo 3D sobre o marcador detectado
                    draw_3d_object_on_opencv(frame, rvecs[0][0], tvecs[0][0])
                    
                    # Desenha o contorno verde (para depuração)
                    cv2.aruco.drawDetectedMarkers(frame, [corners[i]], [ids[i]])

        # Exibe o frame
        cv2.imshow("Realidade Aumentada - OpenCV", frame)
        
        # Sai com a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()