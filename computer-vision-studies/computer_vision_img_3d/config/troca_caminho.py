import os
import shutil
import pandas as pd
from datetime import datetime

# Configurações
CSV_ORIGINAL_PATH = r"C:\Venv\Rep_git\datasets\renders\dataset.csv"
BACKUP_DIR = os.path.dirname(CSV_ORIGINAL_PATH)

# Colunas com caminhos
COLUNAS_CAMINHO = ['rgb_path', 'depth_path', 'stl_path']

# Mapeamento de caminhos antigos para novos
SUBSTITUICOES = {
    r"C:\Venv\OpenCv\datasets\renders": r"C:\Venv\Rep_git\datasets\renders",
    r"C:\Venv\OpenCv\datasets\stl":     r"C:\Venv\Rep_git\datasets\stl"
}

def cria_backup(caminho_arquivo, backup_dir):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    nome_backup = f"dataset_backup_{timestamp}.csv"
    caminho_backup = os.path.join(backup_dir, nome_backup)
    shutil.copy2(caminho_arquivo, caminho_backup)
    print(f"✅ Backup criado em: {caminho_backup}")
    return caminho_backup

def main():
    if not os.path.isfile(CSV_ORIGINAL_PATH):
        print(f"❌ Arquivo CSV não encontrado: {CSV_ORIGINAL_PATH}")
        return

    cria_backup(CSV_ORIGINAL_PATH, BACKUP_DIR)

    df = pd.read_csv(CSV_ORIGINAL_PATH)

    print(f"Colunas encontradas: {list(df.columns)}\n")
    print("🔍 Primeiras linhas do CSV:")
    print(df.head(3), "\n")

    for coluna in COLUNAS_CAMINHO:
        if coluna not in df.columns:
            print(f"⚠️ Coluna '{coluna}' não encontrada — será ignorada.")
            continue

        for antigo, novo in SUBSTITUICOES.items():
            df[coluna] = df[coluna].str.replace(antigo, novo, regex=False)

    nome_novo_csv = os.path.join(BACKUP_DIR, "dataset_modified.csv")
    df.to_csv(nome_novo_csv, index=False)
    print(f"✅ CSV modificado salvo em: {nome_novo_csv}")

if __name__ == "__main__":
    main()
