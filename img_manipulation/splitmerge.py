
import cv2 as cv
import numpy as np

img = cv.imread('Photos/park.jpg')
cv.imshow('Park', img)

#Uma tela preta
blank = np.zeros(img.shape[:2], dtype='uint8')

#Separa os canis de cores
b,g,r = cv.split(img)

# Utiliza o blank para visualiza cada cor separadamente
blue = cv.merge([b,blank,blank])
green = cv.merge([blank,g,blank])
red = cv.merge([blank,blank,r])

cv.imshow('Blue', blue)
cv.imshow('Green', green)
cv.imshow('Red', red)

# Quando usado mostra a tebelade distribuiçã de cada cor - todas são um so tupla
print(img.shape)
print(b.shape)
print(g.shape)
print(r.shape)

# Junta as tonalidades separadas ( B G R )
merged = cv.merge([b,g,r])
cv.imshow('Merged Image', merged)

cv.waitKey(0)