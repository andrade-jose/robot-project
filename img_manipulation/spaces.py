import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('Photos/park.jpg')
cv.imshow('Park', img)

# plt.imshow(img)
# plt.show()


#BGR para grayscale - 
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Cinza', gray)

#BGR para HSV - 
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('HSV', hsv)

#BGR para L*a*b - 
lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
cv.imshow('LAB', lab)

# BGR para RGB
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow('RGB', rgb)

cv.waitKey(0)