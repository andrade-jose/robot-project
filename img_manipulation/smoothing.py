import cv2 as cv

img = cv.imread('Photos/cats.jpg')
cv.imshow('Cats', img)

# Averaging - utiliza a media de cada espaço da matriz karnel
average = cv.blur(img, (3,3))
cv.imshow('Average Blur', average)

# Gaussian Blur - utiliza a media do matriz inteira (tem um desfoque mais leve e natural)
gauss = cv.GaussianBlur(img, (3,3), 0)
cv.imshow('Gaussian Blur', gauss)

# Median Blur - utiliza a mediana de cada pixel da matriz karnel
median = cv.medianBlur(img, 3)
cv.imshow('Median Blur', median)

# Bilateral - utiliza a valores de distacia de centro e area afetada de cada pixel da matriz
bilateral = cv.bilateralFilter(img, 10, 35, 25)
cv.imshow('Bilateral', bilateral)

cv.waitKey(0)