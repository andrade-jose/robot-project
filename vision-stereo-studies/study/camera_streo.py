import numpy as np
import matplotlib.pyplot as plt
# ----- PARÂMETROS DAS DUAS CÂMERAS -----
f = 800  # distância focal em pixels
cx, cy = 320, 240  # centro óptico
baseline = 0.1  # distância entre as duas câmeras (em metros)
K = np.array([[f, 0, cx],
             [0, f, cy],
             [0, 0, 1]])  # matriz intrínseca
# ----- CRIAÇÃO DOS PONTOS 3D NO MUNDO -----
# pontos na frente das câmeras, em diferentes profundidades
depths = np.array([2, 3, 4, 5, 6])  # em metros
points_3D = np.array([[0, 0, z] for z in depths])  # no centro da imagem
# ----- FUNÇÃO DE PROJEÇÃO -----
def project_point(X, Y, Z, tx=0.0):
   # translação da câmera (tx): + para direita
   X_cam = X - tx
   point_camera = np.array([X_cam/Z, Y/Z, 1])
   point_pixel = K @ point_camera
   return point_pixel[:2]
# ----- PROJEÇÃO NAS CÂMERAS -----
points_left = []
points_right = []
for X, Y, Z in points_3D:
   left = project_point(X, Y, Z, tx=0)             # câmera esquerda
   right = project_point(X, Y, Z, tx=baseline)     # câmera direita
   points_left.append(left)
   points_right.append(right)
points_left = np.array(points_left)
points_right = np.array(points_right)
# ----- CÁLCULO DE DISPARIDADE E PROFUNDIDADE -----
disparities = points_left[:, 0] - points_right[:, 0]
estimated_depths = (f * baseline) / disparities
# ----- VISUALIZAÇÃO -----
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].scatter(points_left[:, 0], points_left[:, 1], color='red', label='Câmera Esquerda')
axs[0].scatter(points_right[:, 0], points_right[:, 1], color='blue', label='Câmera Direita')
axs[0].set_title("Projeções Estéreo")
axs[0].invert_yaxis()
axs[0].legend()
axs[0].grid(True)
axs[0].set_aspect('equal')
axs[1].plot(depths, disparities, 'o-')
axs[1].set_title("Disparidade vs Profundidade")
axs[1].set_xlabel("Profundidade Real (m)")
axs[1].set_ylabel("Disparidade (pixels)")
axs[1].grid(True)
plt.tight_layout()
plt.show()
# ----- RESULTADO NUMÉRICO -----
print("Profundidade Real | Disparidade (px) | Profundidade Estimada")
for d, disp, est in zip(depths, disparities, estimated_depths):
   print(f"{d:>7} m         | {disp:>8.2f} px     | {est:>7.2f} m")