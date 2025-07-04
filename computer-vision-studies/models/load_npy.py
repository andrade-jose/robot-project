import numpy as np

# Caminho correto para o arquivo .npy
arquivo = r"C:\Venv\OpenCv\computer-vision-studies\models\logs\history_advanced_scratch_20250628_083332.npy"

# Carrega com pickle permitido
dados = np.load(arquivo, allow_pickle=True)

print("Tipo:", type(dados))
print("Conteúdo:")
print(dados)

