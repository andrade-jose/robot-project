from pathlib import Path
import re

def renomear_exr_depth(render_root):
    root = Path(render_root)

    for subpasta in root.iterdir():
        if not subpasta.is_dir():
            continue

        for arquivo in subpasta.glob("*.exr"):
            # Regex para encontrar '_depth' + qualquer coisa antes da extensão
            novo_nome = re.sub(r"(_depth).*\.exr$", r"\1.exr", arquivo.name)

            if arquivo.name != novo_nome:
                novo_caminho = arquivo.with_name(novo_nome)
                
                if novo_caminho.exists():
                    print(f"Aviso: arquivo {novo_caminho} já existe. Ignorando renomeação de {arquivo.name}")
                    continue
                
                print(f"Renomeando {arquivo.name} → {novo_nome} em {subpasta.name}")
                arquivo.rename(novo_caminho)

if __name__ == "__main__":
    renomear_exr_depth(r"C:\Venv\OpenCv\datasets\renders")
