import pandas as pd
import os
import shutil
from glob import glob

# === CONFIGURAÇÕES ===
pasta_csvs = r"C:\Venv\Robot_project\robot-project\vision-stereo-studies\camera_project\datasets\dados"
pasta_imagens = r"C:\Venv\Robot_project\robot-project\vision-stereo-studies\camera_project\datasets\images"
novo_dir = r"C:\Venv\Robot_project\robot-project\vision-stereo-studies\camera_project\best_img"
top_n = 60
coluna_imagem = 'Imagem'
coluna_qualidade = 'Qualidade_Total'

# Criar pastas de saída
os.makedirs(os.path.join(novo_dir, 'left'), exist_ok=True)
os.makedirs(os.path.join(novo_dir, 'right'), exist_ok=True)

# Procurar todos os CSVs
csv_paths = glob(os.path.join(pasta_csvs, '*.csv'))

if not csv_paths:
    print("⚠️ Nenhum CSV encontrado na pasta:", pasta_csvs)
    exit()

# Lê todos os CSVs e monta dataframe único
dataframes = []
for csv_path in csv_paths:
    nome_base = os.path.splitext(os.path.basename(csv_path))[0]  # seq01.csv → seq01
    dir_left = os.path.join(pasta_imagens, nome_base, 'left')
    dir_right = os.path.join(pasta_imagens, nome_base, 'right')
    print(f'--- Verificando CSV: {csv_path}')
    print(f'    Espera imagens em: {dir_left}')
    print(f'                      e: {dir_right}')


    if not os.path.isdir(dir_left) or not os.path.isdir(dir_right):
        print(f'❌ Diretórios left/right não encontrados para: {nome_base}')
        continue

    df = pd.read_csv(csv_path)
    df['seq'] = nome_base
    df['left_path'] = df[coluna_imagem].apply(lambda img: os.path.join(dir_left, img))
    df['right_path'] = df[coluna_imagem].apply(lambda img: os.path.join(dir_right, img))
    dataframes.append(df)

if not dataframes:
    print("⚠️ Nenhum par CSV + imagens válido encontrado.")
    exit()

# Unir e ordenar
df_total = pd.concat(dataframes, ignore_index=True)
df_top = df_total.sort_values(by=coluna_qualidade, ascending=False).head(top_n)

# Gerar nomes novos: img_0001.png, img_0002.png, ...
novos_nomes = [f'img_{i:04d}.png' for i in range(1, len(df_top) + 1)]
df_top = df_top.reset_index(drop=True)
df_top['novo_nome'] = novos_nomes

# Copiar arquivos com novos nomes
for i, row in df_top.iterrows():
    novo_nome = row['novo_nome']
    
    # LEFT
    if os.path.exists(row['left_path']):
        shutil.copy2(row['left_path'], os.path.join(novo_dir, 'left', novo_nome))
        print(f'✅ Left:  {novo_nome}')
    else:
        print(f'⚠️ LEFT não encontrado: {row["left_path"]}')
    
    # RIGHT
    if os.path.exists(row['right_path']):
        shutil.copy2(row['right_path'], os.path.join(novo_dir, 'right', novo_nome))
        print(f'✅ Right: {novo_nome}')
    else:
        print(f'⚠️ RIGHT não encontrado: {row["right_path"]}')

# Salvar CSV com metadados
colunas_exportar = [
    'novo_nome', coluna_imagem, coluna_qualidade, 'seq', 'left_path', 'right_path'
]
df_top[colunas_exportar].to_csv(os.path.join(novo_dir, 'melhores.csv'), index=False)

print(f'\n📄 CSV salvo com nomes renomeados: {os.path.join(novo_dir, "melhores.csv")}')
print(f'🎉 Processo completo. {top_n} pares copiados e renomeados.')
