import os
import cv2 as cv
import numpy as np

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
DIR = r'C:\Venv\OpenCv\Faces\train'

# Colocar o xml na variavel
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# Listas par rostos e nomes
faetures = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        # Loop para correr o diretorio e procura rostos
        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            # Detecta o rosto utilizando o xml - variais de escala
            # vizinhos (pode ser usado para diminuir o ruido - amenizar e erro de decteção)
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            # um loop para mapear rotso e nomes
            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                faetures.append(faces_roi)
                labels.append(label)

create_train()

print('Treino terminado--------------------------')
faetures = np.array(faetures, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Treina o reconhecimento com a lista de rosto e lista de nome
face_recognizer.train(faetures,labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy',faetures)
np.save('labels.npy', labels)
