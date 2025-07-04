import os
import random
import json
from pathlib import Path
from OCC.Core.BRepPrimAPI import (
    BRepPrimAPI_MakeBox,
    BRepPrimAPI_MakeSphere,
    BRepPrimAPI_MakeCylinder,
    BRepPrimAPI_MakeCone,
    BRepPrimAPI_MakeTorus,
)
from OCC.Core.gp import gp_Pnt
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakePolygon,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_Sewing,
    BRepBuilderAPI_MakeSolid,
)
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs

IMAGES_PER_SHAPE = 250
SCALE_FACTOR = 0.01 
SHAPES = ["cube", "sphere", "cylinder", "cone", "torus", "pyramid", "prism"]

OUTPUT_DIR = Path("C:/Venv/OpenCv/computer-vision-studies/datasets/STEP_blender")

def make_pyramid(base_size, height):
    p1 = gp_Pnt(0, 0, 0)
    p2 = gp_Pnt(base_size, 0, 0)
    p3 = gp_Pnt(base_size / 2, base_size * 0.866, 0)
    apex = gp_Pnt(base_size / 2, base_size * 0.2887, height)

    polygon = BRepBuilderAPI_MakePolygon()
    for pt in [p1, p2, p3, p1]:
        polygon.Add(pt)
    base_wire = polygon.Wire()
    base_face = BRepBuilderAPI_MakeFace(base_wire).Face()

    edges = [(p1, p2), (p2, p3), (p3, p1)]
    side_faces = []
    for start, end in edges:
        poly_side = BRepBuilderAPI_MakePolygon()
        poly_side.Add(start)
        poly_side.Add(end)
        poly_side.Add(apex)
        poly_side.Add(start)
        wire_side = poly_side.Wire()
        face_side = BRepBuilderAPI_MakeFace(wire_side).Face()
        side_faces.append(face_side)

    sewing = BRepBuilderAPI_Sewing()
    sewing.Add(base_face)
    for face in side_faces:
        sewing.Add(face)
    sewing.Perform()
    shell = sewing.SewedShape()

    solid_maker = BRepBuilderAPI_MakeSolid()
    solid_maker.Add(shell)
    if not solid_maker.IsDone():
        raise RuntimeError("Falha ao criar sólido a partir do shell.")
    return solid_maker.Solid()

def make_prism(base_size, height):
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
    from OCC.Core.gp import gp_Vec

    p1 = gp_Pnt(0, 0, 0)
    p2 = gp_Pnt(base_size, 0, 0)
    p3 = gp_Pnt(base_size / 2, base_size * 0.866, 0)

    polygon = BRepBuilderAPI_MakePolygon()
    for pt in [p1, p2, p3, p1]:
        polygon.Add(pt)
    wire = polygon.Wire()

    face = BRepBuilderAPI_MakeFace(wire).Face()
    vec = gp_Vec(0, 0, height)
    prism = BRepPrimAPI_MakePrism(face, vec).Shape()
    return prism

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
            elif shape_type == "sphere":
                params = [random.uniform(5, 15) * SCALE_FACTOR]
                shape = BRepPrimAPI_MakeSphere(params[0]).Shape()
            elif shape_type == "cylinder":
                params = [random.uniform(3, 10) * SCALE_FACTOR, random.uniform(5, 20) * SCALE_FACTOR]
                shape = BRepPrimAPI_MakeCylinder(*params).Shape()
            elif shape_type == "cone":
                params = [random.uniform(3, 10) * SCALE_FACTOR, random.uniform(5, 20) * SCALE_FACTOR]
                shape = BRepPrimAPI_MakeCone(params[0], 0.0, params[1]).Shape()
            elif shape_type == "torus":
                params = [random.uniform(5, 15) * SCALE_FACTOR, random.uniform(1, 5) * SCALE_FACTOR]
                shape = BRepPrimAPI_MakeTorus(params[0], params[1]).Shape()
            elif shape_type == "pyramid":
                params = [random.uniform(5, 15) * SCALE_FACTOR, random.uniform(5, 20) * SCALE_FACTOR]
                shape = make_pyramid(params[0], params[1])
            elif shape_type == "prism":
                params = [random.uniform(5, 15) * SCALE_FACTOR, random.uniform(5, 20) * SCALE_FACTOR]
                shape = make_prism(params[0], params[1])
            else:
                continue


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
