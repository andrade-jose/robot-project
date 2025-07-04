from pathlib import Path
import json

def renomear_depth_exr_para_nome_correto(render_root):
    root = Path(render_root)

    for subpasta in root.iterdir():
        if not subpasta.is_dir():
            continue

        depth_exr_path = subpasta / "depth.exr"
        if not depth_exr_path.exists():
            print(f"Aviso: depth.exr não encontrado em {subpasta.name}")
            continue

        views_info_path = subpasta / "views_info.json"
        if not views_info_path.exists():
            print(f"Aviso: views_info.json não encontrado em {subpasta.name}")
            continue

        try:
            with open(views_info_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Erro ao ler {views_info_path}: {e}")
            continue

        # Verifica se data é lista e tem pelo menos um item
        if not isinstance(data, list) or len(data) == 0:
            print(f"Formato inesperado no JSON em {subpasta.name}")
            continue

        nome_correto_exr = Path(data[0].get("depth_path", "")).name
        if not nome_correto_exr:
            print(f"Nome correto do .exr não encontrado no JSON em {subpasta.name}")
            continue

        novo_caminho = subpasta / nome_correto_exr

        if novo_caminho.exists():
            print(f"Aviso: arquivo {novo_caminho} já existe. Ignorando renomeação em {subpasta.name}")
            continue

        print(f"Renomeando depth.exr → {nome_correto_exr} em {subpasta.name}")
        depth_exr_path.rename(novo_caminho)

if __name__ == "__main__":
    renomear_depth_exr_para_nome_correto(r"C:\Venv\OpenCv\datasets\renders")
