import cv2 as cv

img = cv.imread('Photos/park.jpg')
cv.imshow('Park', img)

# Converte as cores para tons de cinza
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Cinza', gray)

# Bora a imagem - ksize tem que ser sempre impar
blur = cv.blur(img,(3,3),cv.BORDER_DEFAULT)
cv.imshow('Borado', blur)

# Encontrar bordas das imagens
'''
E muito util usar junto ou blur pois reduz a qualidade da imagem,
assim podemos dar o enfoque em um item essencial e pegar suas bordas.
'''
canny = cv.Canny(img, 125, 175)
cv.imshow('Bordas', canny)

# Dilatar a imagem - pode se dizer que estica os pixels da imagem
dilated = cv.dilate(img, (3,3), iterations=3)
cv.imshow('Dilatada', dilated)

# Erodir a imagem - de maneira informal encolhe os pixels
eroded = cv.erode(img, (3,3), iterations=3)
cv.imshow('Erodida', eroded)

# Redimecionar a imagem - para manter usar INTER - CUBIC(lento, melhor qld), LINEAR(ampliada) OU AREA(reduzida)
resized = cv.resize(img, (500,500), cv.INTER_CUBIC)
cv.imshow('Redimencionada', resized)

# Recortar a imagem tamanho e lugar
cropped = img[50:200, 200:400]
cv.imshow('Recortada', cropped)


cv.waitKey(0)