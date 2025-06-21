import cv2 as cv

#REDIMENSIONAR - util para Imagens, Videos, Lives
def rescaleFrame(frame, escala = 0.75):
  largura  = int(frame.shape[1] * escala)
  altura = int(frame.shape[0]*escala)

  dimensoes = (largura,altura)

  return cv.resize(frame, dimensoes, interpolation=cv.INTER_AREA)

#REDIMENSIONAR - util somente para alterar lives
def chengeRes(largura,altura):
  capture.set(3,largura)
  capture.set(4,altura)
  

img = cv.imread('Photos/cat.jpg')
cv.imshow('Cat', img)

resized_imagem = rescaleFrame(img)
cv.imshow('Imagem', resized_imagem)

capture = cv.VideoCapture('Videos/dog.mp4')

while True:
   isTrue, frame = capture.read()

   frame_resized = rescaleFrame(frame, escala=.2)

   cv.imshow('Video', frame)
   cv.imshow('Video Resized', frame_resized)
   
   if cv.waitKey(20) & 0xFF==ord('d'):
    break

capture.release()
cv.destroyAllWindows()

