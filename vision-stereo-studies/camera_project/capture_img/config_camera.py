import cv2

# Abre a câmera
cap = cv2.VideoCapture(0)

# Define função para mostrar valores
def print_values():
    print("\n--- Parâmetros da Câmera ---")
    print(f"Brilho     : {cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
    print(f"Contraste  : {cap.get(cv2.CAP_PROP_CONTRAST)}")
    print(f"Saturação  : {cap.get(cv2.CAP_PROP_SATURATION)}")
    print(f"Nitidez (*) : {cap.get(cv2.CAP_PROP_SHARPNESS) if hasattr(cv2, 'CAP_PROP_SHARPNESS') else 'Não disponível'}")

# Valores iniciais
brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
contrast   = cap.get(cv2.CAP_PROP_CONTRAST)
saturation = cap.get(cv2.CAP_PROP_SATURATION)
sharpness  = 0  # nem toda câmera permite

print_values()

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERRO] Não conseguiu capturar imagem")
        break

    cv2.imshow("Ajuste da Câmera", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):  # Quit
        break

    # Ajustes com teclas
    elif key == ord('w'):  # Brilho +
        brightness += 1
        cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
    elif key == ord('s'):  # Brilho -
        brightness -= 1
        cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)

    elif key == ord('e'):  # Contraste +
        contrast += 1
        cap.set(cv2.CAP_PROP_CONTRAST, contrast)
    elif key == ord('d'):  # Contraste -
        contrast -= 1
        cap.set(cv2.CAP_PROP_CONTRAST, contrast)

    elif key == ord('r'):  # Saturação +
        saturation += 1
        cap.set(cv2.CAP_PROP_SATURATION, saturation)
    elif key == ord('f'):  # Saturação -
        saturation -= 1
        cap.set(cv2.CAP_PROP_SATURATION, saturation)

    elif key == ord('p'):  # Mostrar valores
        print_values()

cv2.destroyAllWindows()
cap.release()
