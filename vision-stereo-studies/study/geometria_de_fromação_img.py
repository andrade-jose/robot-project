import numpy as np

import matplotlib.pyplot as plt

# ---- PARTE 1: pontos 3D no mundo (X, Y, Z) ----

points_3D = np.array([

    [-1,  1, 4],

    [ 0,  1, 4],

    [ 1,  1, 4],

    [-1,  0, 4],

    [ 0,  0, 4],

    [ 1,  0, 4],

    [-1, -1, 4],

    [ 0, -1, 4],

    [ 1, -1, 4]

])  # pontos em uma grade 3x3 a Z=4m

# ---- PARTE 2: parâmetros da câmera ----

f = 800  # distância focal (em pixels)

cx, cy = 320, 240  # centro óptico da imagem

K = np.array([[f, 0, cx],

              [0, f, cy],

              [0, 0, 1]])  # matriz intrínseca

# ---- PARTE 3: projeção de cada ponto 3D para 2D ----

points_2D = []

for X, Y, Z in points_3D:

    point_camera = np.array([X/Z, Y/Z, 1])  # coordenada normalizada

    point_pixel = K @ point_camera

    x, y = point_pixel[0], point_pixel[1]

    points_2D.append([x, y])

points_2D = np.array(points_2D)

# ---- PARTE 4: exibição ----

plt.figure(figsize=(6, 5))

plt.scatter(points_2D[:, 0], points_2D[:, 1], color='red')

plt.title("Projeção de Pontos 3D no Plano da Imagem")

plt.xlabel("x (pixels)")

plt.ylabel("y (pixels)")

plt.xlim(0, 640)

plt.ylim(480, 0)

plt.grid(True)

plt.gca().set_aspect('equal')

plt.show()
 