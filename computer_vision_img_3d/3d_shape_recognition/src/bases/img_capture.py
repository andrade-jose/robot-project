import cv2
import numpy as np

cap = cv2.VideoCapture(0)  # Use 0 para webcam interna

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mostrar a imagem ao vivo
    cv2.imshow("Camera", frame)

    # Preparar imagem e prever
    input_img = prepare(frame)
    pred = model.predict(input_img)
    label = characters[np.argmax(pred[0])]

    print(f"Predição: {label}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
