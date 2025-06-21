import cv2 as cv

img = cv.imread('Photos/group 1.jpg')
cv.imshow('Grupo de pessoas', img)

# A castaca não considera cor - pode até atrapalhar
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Colocar o xml na variavel
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# Detecta o rosto utilizando o xml - variais de escala
# vizinhos (pode ser usado para diminuir o ruido - amenizar e erro de decteção)
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

print(f'Numeros de rotos encontrados = {len(faces_rect)}')

# um loop que colocar um retangulo em cada rosto que foi dectado
for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected face', img)

cv.waitKey(0)