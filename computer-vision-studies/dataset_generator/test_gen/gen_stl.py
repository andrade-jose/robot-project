import os
import subprocess
import json
import random
import sys
from OCC.Core.BRepPrimAPI import (
    BRepPrimAPI_MakeBox,
    BRepPrimAPI_MakeSphere,
    BRepPrimAPI_MakeCylinder,
    BRepPrimAPI_MakeCone,
    BRepPrimAPI_MakeTorus,
)
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs

IMAGES_PER_SHAPE = 250
SHAPES = ["cube", "sphere", "cylinder", "cone", "torus", "pyramid", "prism"]
OUTPUT_DIR = "C:/Venv/OpenCv/computer-vision-studies/datasets/STEP_blender"

from OCC.Core.gp import gp_Pnt
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakePolygon,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_Sewing,
    BRepBuilderAPI_MakeSolid,
)

def make_pyramid(base_size, height):
    # Vértices da base triangular equilátera
    p1 = gp_Pnt(0, 0, 0)
    p2 = gp_Pnt(base_size, 0, 0)
    p3 = gp_Pnt(base_size / 2, base_size * 0.866, 0)  # altura tri equilátero: base_size * sqrt(3)/2

    # Ápice da pirâmide (posição vertical)
    apex = gp_Pnt(base_size / 2, base_size * 0.2887, height)  # posição dentro da base

    # Base da pirâmide (triangular)
    polygon = BRepBuilderAPI_MakePolygon()
    for pt in [p1, p2, p3, p1]:  # fecha o polígono
        polygon.Add(pt)
    base_wire = polygon.Wire()
    base_face = BRepBuilderAPI_MakeFace(base_wire).Face()

    # Faces laterais (triângulos que conectam arestas da base ao ápice)
    edges = [(p1, p2), (p2, p3), (p3, p1)]
    side_faces = []
    for start, end in edges:
        poly_side = BRepBuilderAPI_MakePolygon()
        poly_side.Add(start)
        poly_side.Add(end)
        poly_side.Add(apex)
        poly_side.Add(start)  # fecha o polígono lateral
        wire_side = poly_side.Wire()
        face_side = BRepBuilderAPI_MakeFace(wire_side).Face()
        side_faces.append(face_side)

    # Costura todas as faces para formar um shell fechado
    sewing = BRepBuilderAPI_Sewing()
    sewing.Add(base_face)
    for face in side_faces:
        sewing.Add(face)
    sewing.Perform()
    shell = sewing.SewedShape()

    # Cria sólido a partir do shell
    solid_maker = BRepBuilderAPI_MakeSolid()
    solid_maker.Add(shell)

    if not solid_maker.IsDone():
        raise RuntimeError("Falha ao criar sólido a partir do shell.")

    return solid_maker.Solid()

def make_prism(base_size, height):
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
    from OCC.Core.gp import gp_Vec

    # Define pontos do triângulo base
    p1 = gp_Pnt(0, 0, 0)
    p2 = gp_Pnt(base_size, 0, 0)
    p3 = gp_Pnt(base_size / 2, base_size * 0.866, 0)

    polygon = BRepBuilderAPI_MakePolygon()
    for pt in [p1, p2, p3, p1]:
        polygon.Add(pt)
    wire = polygon.Wire()

    face = BRepBuilderAPI_MakeFace(wire).Face()
    vec = gp_Vec(0, 0, height)  # vetor de extrusão (altura)

    prism = BRepPrimAPI_MakePrism(face, vec).Shape()
    return prism


def save_step(shape, filename):
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    status = writer.Write(filename)
    if status != 1:
        print(f"[Erro] Não foi possível salvar STEP: {filename}")

def generate_shapes(output_dir):
    import json
    os.makedirs(os.path.join(output_dir, "step"), exist_ok=True)
    metadata = {}

    for shape_type in SHAPES:
        for i in range(IMAGES_PER_SHAPE):
            if shape_type == "cube":
                params = [random.uniform(5, 20) for _ in range(3)]
                shape = BRepPrimAPI_MakeBox(*params).Shape()
            elif shape_type == "sphere":
                params = [random.uniform(5, 15)]
                shape = BRepPrimAPI_MakeSphere(params[0]).Shape()
            elif shape_type == "cylinder":
                params = [random.uniform(3, 10), random.uniform(5, 20)]
                shape = BRepPrimAPI_MakeCylinder(*params).Shape()
            elif shape_type == "cone":
                params = [random.uniform(3, 10), random.uniform(5, 20)]
                shape = BRepPrimAPI_MakeCone(params[0], 0.0, params[1]).Shape()
            elif shape_type == "torus":
                params = [random.uniform(5, 15), random.uniform(1, 5)]
                shape = BRepPrimAPI_MakeTorus(params[0], params[1]).Shape()
            elif shape_type == "pyramid":
                params = [random.uniform(5, 15), random.uniform(5, 20)]
                shape = make_pyramid(params[0], params[1])
            elif shape_type == "prism":
                params = [random.uniform(5, 15), random.uniform(5, 20)]
                shape = make_prism(params[0], params[1])
            else:
                continue

            base_filename = f"{shape_type}_{i+1:04d}"
            step_path = os.path.join(output_dir, "step", base_filename + ".step")

            save_step(shape, step_path)

            metadata[base_filename] = {
                "type": shape_type,
                "parameters": params,
                "paths": {
                    "step": step_path,
                    "stl": os.path.join(output_dir, "stl", base_filename + ".stl"),
                    "render": os.path.join(output_dir, "images", base_filename + ".png")
                }
            }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    output_dir = r"C:\Venv\OpenCv\computer-vision-studies\datasets\STEP_blender\step"
    variants_per_shape = 250
    generate_shapes(output_dir, variants_per_shape)
