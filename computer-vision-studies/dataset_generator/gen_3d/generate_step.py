import os
import random
import json
from pathlib import Path
from OCC.Core.BRepPrimAPI import (
    BRepPrimAPI_MakeBox,
    BRepPrimAPI_MakeCylinder,
)
from OCC.Core.gp import gp_Pnt
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs

IMAGES_PER_SHAPE = 500
SCALE_FACTOR = 0.02
SHAPES = ["cube","cylinder"]

OUTPUT_DIR = Path("C:\Venv\Rep_git\datasets\step_2")

def save_step(shape, filename):
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    status = writer.Write(str(filename))
    if status != 1:
        print(f"[Erro] Não foi possível salvar STEP: {filename}")

def generate_shapes(output_dir):
    step_dir = output_dir / "step"
    step_dir.mkdir(parents=True, exist_ok=True)
    metadata = {}

    for shape_type in SHAPES:
        for i in range(IMAGES_PER_SHAPE):
            if shape_type == "cube":
                params = [random.uniform(5, 20) * SCALE_FACTOR for _ in range(3)]
                shape = BRepPrimAPI_MakeBox(*params).Shape()
            elif shape_type == "cylinder":
                params = [random.uniform(3, 10) * SCALE_FACTOR, random.uniform(5, 20) * SCALE_FACTOR]
                shape = BRepPrimAPI_MakeCylinder(*params).Shape()


            base_filename = f"{shape_type}_{i+1:04d}"
            step_path = step_dir / f"{base_filename}.step"

            save_step(shape, step_path)

            metadata[base_filename] = {
                "type": shape_type,
                "parameters": params,
                "paths": {
                    "step": str(step_path),
                    "stl": str(output_dir / "stl" / f"{base_filename}.stl"),
                    "render": str(output_dir / "images" / f"{base_filename}.png"),
                },
            }

    # Salva metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    print("=== Gerando arquivos STEP ===")
    generate_shapes(OUTPUT_DIR)
    print("=== Arquivos STEP gerados em:", OUTPUT_DIR / "step")
