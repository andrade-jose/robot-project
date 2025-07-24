from pathlib import Path
import json
import csv

def encontrar_json(pasta: Path):
    """Retorna o primeiro arquivo .json encontrado na pasta."""
    return next(pasta.glob("*.json"), None)

def carregar_views(json_path: Path):
    try:
        with json_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Erro ao ler {json_path}: {e}")
        return []

def construir_linha(view, subpasta: Path, model_name: str, shape_type: str, stl_path: Path):
    return {
        "model_name": model_name,
        "shape_type": shape_type,
        "view_idx": view.get("view_idx", ""),
        "rgb_path": str(subpasta / Path(view.get("rgb_path", "")).name),
        "depth_path": str(subpasta / Path(view.get("depth_path", "")).name),
        "camera_angle": view.get("camera_angle", ""),
        "camera_height": view.get("camera_height", ""),
        "background_color": json.dumps(view.get("background_color", [])),
        "material_color": json.dumps(view.get("material_color", [])),
        "stl_path": str(stl_path) if stl_path.exists() else ""
    }

def coletar_jsons_e_gerar_csv(render_root, stl_dir, csv_output_path):
    render_root = Path(render_root)
    stl_dir = Path(stl_dir)
    csv_output_path = Path(csv_output_path)

    rows = []

    for subpasta in sorted(render_root.iterdir()):
        if not subpasta.is_dir():
            continue

        model_name = subpasta.name
        shape_type = model_name.split("_")[0]

        json_path = encontrar_json(subpasta)
        if not json_path:
            print(f"Aviso: Nenhum JSON encontrado em {model_name}")
            continue

        views = carregar_views(json_path)
        if not views:
            continue

        stl_path = stl_dir / f"{model_name}.stl"
        if not stl_path.exists():
            print(f"Aviso: STL não encontrado para {model_name}")

        for view in views:
            row = construir_linha(view, subpasta, model_name, shape_type, stl_path)
            rows.append(row)

    if not rows:
        raise RuntimeError("Nenhum dado foi extraído dos JSONs.")

    # Salvar CSV
    with csv_output_path.open('w', newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ CSV gerado com {len(rows)} entradas: {csv_output_path}")

# Exemplo de uso
if __name__ == "__main__":
    render_root = r"C:\Venv\Rep_git\datasets\dataset_tapatan\renders"
    stl_dir = r"C:\Venv\Rep_git\datasets\dataset_tapatan\stl"
    csv_out = r"C:\Venv\Rep_git\datasets\dataset_tapatan\dataset.csv"

    coletar_jsons_e_gerar_csv(render_root, stl_dir, csv_out)
