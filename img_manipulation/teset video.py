import cv2 as cv


capture = cv.VideoCapture('Videos/teste 2.mp4')

while True:
    isTrue, frame = capture.read()

    #cv.imshow('Video', frame)

    # A castaca não considera cor - pode até atrapalhar
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('Gray', gray)

    # Colocar o xml na variavel
    haar_cascade = cv.CascadeClassifier('haar_face.xml')

    # Detecta o rosto utilizando o xml - variais de escala
    # vizinhos (pode ser usado para diminuir o ruido - amenizar e erro de decteção)
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

    #print(f'Numeros de rotos encontrados = {len(faces_rect)}')

    # um loop que colocar um retangulo em cada rosto que foi dectado
    for (x,y,w,h) in faces_rect:
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)

    cv.imshow('Detected face', frame)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()