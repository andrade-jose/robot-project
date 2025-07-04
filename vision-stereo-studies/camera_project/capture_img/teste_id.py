import cv2

cam_id = 1  # troque para 1, 2, etc. e teste

cap = cv2.VideoCapture(cam_id)
if not cap.isOpened():
    print(f"Não foi possível abrir a câmera com ID {cam_id}")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow(f"Câmera {cam_id}", frame)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
