import pandas as pd
import os
import sys
from pathlib import Path

# Configuração do caminho do projeto
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


from config.config import LEFT_DIR2,RIGHT_DIR2, CSV_FILE2
# Configurações
csv_path = CSV_FILE2       # caminho para o seu arquivo CSV
image_dir_left = LEFT_DIR2         # diretório onde estão as imagens
image_dir_rigth = RIGHT_DIR2       # diretório onde estão as imagens
coluna_imagem = 'Imagem'       # nome da coluna no CSV com os nomes das imagens
coluna_qualidade = 'Qualidade_Total' # nome da coluna que contém o valor da qualidade

# Passo 1: Ler CSV
df = pd.read_csv(csv_path)

# Passo 2: Ordenar pelo valor da qualidade, descrescente, e pegar as top 30 imagens
top_30 = df.sort_values(by=coluna_qualidade, ascending=False).head(30)

# Criar um conjunto com os nomes das imagens que queremos manter
imagens_que_ficam = set(top_30[coluna_imagem].tolist())

# Passo 3: Listar todas as imagens do diretório e apagar as que não estão no top 30
for filename in os.listdir(image_dir_left):
    if filename not in imagens_que_ficam:
        caminho = os.path.join(image_dir_left, filename)
        if os.path.isfile(caminho):
            os.remove(caminho)
            print(f'Arquivo removido: {filename}')

# Passo 3: Listar todas as imagens do diretório e apagar as que não estão no top 30
for filename in os.listdir(image_dir_rigth):
    if filename not in imagens_que_ficam:
        caminho = os.path.join(image_dir_rigth, filename)
        if os.path.isfile(caminho):
            os.remove(caminho)
            print(f'Arquivo removido: {filename}')


print("Processo finalizado.")
