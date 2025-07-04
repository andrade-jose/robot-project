import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model
import caer

# --- Parâmetros ---
IMG_SIZE = (80, 80)
CHANNELS = 1

# --- Lista das classes ---
characters = ['cubo', 'esfera', 'cone']

# --- Carregar modelo ---
model = load_model('best_model.h5')

def prepare(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    resized = cv.resize(gray, IMG_SIZE)
    equalized = cv.equalizeHist(resized)
    reshaped = caer.reshape(equalized, IMG_SIZE, CHANNELS)
    reshaped = reshaped.astype('float32') / 255.0
    return reshaped.reshape(1, IMG_SIZE[0], IMG_SIZE[1], CHANNELS), equalized  # Retorna tanto a versão para o modelo quanto para visualização

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Pré-processamento (agora retorna ambas as versões)
    input_img, processed_img = prepare(frame)
    
    # Predição
    prediction = model.predict(input_img, verbose=0)[0]
    label_idx = np.argmax(prediction)
    label = characters[label_idx]
    confidence = np.max(prediction) * 100
    
    # Exibir informações
    cv.putText(frame, f"{label} ({confidence:.1f}%)", (10, 30), 
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Mostrar imagem processada CORRETAMENTE
    # Já temos a versão equalizada (processed_img) que é uint8
    processed_display = cv.resize(processed_img, (200, 200))
    cv.imshow("Imagem Processada", processed_display)
    
    cv.imshow("Reconhecimento 3D", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
cap.release()
cv.destroyAllWindows()