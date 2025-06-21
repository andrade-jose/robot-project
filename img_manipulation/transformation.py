import cv2 as cv
import numpy as np
img = cv.imread('Photos/park.jpg')
cv.imshow('Park', img)

#Transformação
def translate(img, x, y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)
# -x --> esquerda
# -y --> cima
#  x --> direita
#  y --> baixo

translate = translate(img, -100, 100)
cv.imshow('tranformado', translate)

# Rodar a imagem - repetir o processo de rotação torna um parte preta.

def rotate(img, angle, rotPoint=None):
    (altura,largura) = img.shape[:2]

    if rotPoint is None:
        rotPoint = (largura//2,altura//2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (largura,altura)

    return cv.warpAffine(img, rotMat, dimensions)
        
rotated = rotate(img, 45)
cv.imshow('Rotacionada', rotated)

rotated_rotaded = rotate(rotated, 45)
cv.imshow('Re-rotacionado', rotated_rotaded)

resized = cv.resize(img, (500,500), cv.INTER_CUBIC)
cv.imshow('Redimensionar', resized)

# Virar a imagem
flip = cv.flip(img, 1)
cv.imshow('Virada', flip)


cv.waitKey(0)
