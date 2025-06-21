import cv2 as cv
import numpy as np

blank = np.zeros((500,500,3), dtype='uint8')
cv.imshow('Blank', blank)

# 1 definar um cor para imagem
blank[:] = 0,255,0
cv.imshow('Green', blank)

#1.2 definir uma cor em um espaço determinado
blank[200:300, 300:400] = 0,0,255
cv.imshow('Green', blank)

#2 desenhando um retangulo (  formato pode ser escolhido )
cv.rectangle(blank, (250,250), (500,500), (0,255,0), thickness=2)
cv.imshow('Retangulo', blank)

# Outra forma de fazer o formato
cv.rectangle(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2), (0,255,0), thickness=-1)
cv.imshow('Retangulo', blank)

#2.2 como preencher o espaço do 'retangulo' (FILLED) OU (-1)
cv.rectangle(blank, (0,0), (250,250), (0,255,0), thickness=cv.FILLED)
cv.imshow('Retangulo', blank)

# 3 desenhar um circulo
cv.circle(blank,  (blank.shape[1]//2, blank.shape[0]//2), 50, (0,0,255), thickness=5)
cv.imshow('Circulo', blank)

# 4 desenhar um linha
cv.line(blank,(100,250),(300,400),(255,255,255), 3 )
cv.imshow('Linha',blank)

# Escrever texto
cv.putText(blank, 'Ola Teste', (0,225), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0,255,0), 4)
cv.imshow('Texto', blank)

cv.waitKey(0)
