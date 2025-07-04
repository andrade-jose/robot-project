import os
import random
import json
import subprocess
from OCC.Core.BRepPrimAPI import (
    BRepPrimAPI_MakeBox, BRepPrimAPI_MakeSphere,
    BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeCone,
    BRepPrimAPI_MakeTorus
)
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakePolygon
from OCC.Core.gp import gp_Pnt
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.StlAPI import StlAPI_Writer

# ===== CONFIGURAÇÕES =====
OUTPUT_DIR = "dataset_3d_shapes"
RENDER_RESOLUTION = (512, 512)
IMAGES_PER_SHAPE = 250  # Para 8 formas, 250x8=2000 imagens
SHAPES = ["cube", "sphere", "cylinder", "cone", "torus", "pyramid", "prism"]

# ===== 1. GERAÇÃO DAS FORMAS =====
def generate_shapes():
    os.makedirs(os.path.join(OUTPUT_DIR, "step"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "stl"), exist_ok=True)
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
                # toro: raio maior, raio do tubo
                params = [random.uniform(5, 15), random.uniform(1, 5)]
                shape = BRepPrimAPI_MakeTorus(params[0], params[1]).Shape()
            elif shape_type == "pyramid":
                # Pirâmide triangular: base e altura
                params = [random.uniform(5, 15), random.uniform(5, 20)]
                shape = make_pyramid(params[0], params[1])
            elif shape_type == "prism":
                # Prisma triangular: base lado e altura
                params = [random.uniform(5, 15), random.uniform(5, 20)]
                shape = make_prism(params[0], params[1])
            else:
                continue

            base_filename = f"{shape_type}_{i+1:04d}"
            step_path = os.path.join(OUTPUT_DIR, "step", base_filename + ".step")
            stl_path = os.path.join(OUTPUT_DIR, "stl", base_filename + ".stl")

            save_step(shape, step_path)
            save_stl(shape, stl_path)

            metadata[base_filename] = {
                "type": shape_type,
                "parameters": params,
                "paths": {
                    "step": step_path,
                    "stl": stl_path,
                    "render": os.path.join(OUTPUT_DIR, "images", base_filename + ".png")
                }
            }

    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def save_step(shape, filename):
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    status = writer.Write(filename)
    if status != 1:
        print(f"[Erro] Não foi possível salvar STEP: {filename}")


def save_stl(shape, filename):
    writer = StlAPI_Writer()
    writer.Write(shape, filename)


# ===== FUNÇÕES AUXILIARES DE FORMAS COMPLEXAS =====
def make_pyramid(base_size, height):
    # Pirâmide triangular simples
    p1 = gp_Pnt(0, 0, 0)
    p2 = gp_Pnt(base_size, 0, 0)
    p3 = gp_Pnt(base_size / 2, base_size * 0.866, 0)  # base triangular equilátera
    apex = gp_Pnt(base_size / 2, base_size * 0.2887, height)

    polygon = BRepBuilderAPI_MakePolygon()
    for pt in [p1, p2, p3, p1]:
        polygon.Add(pt)
    wire = polygon.Wire()

    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    base_face = BRepBuilderAPI_MakeFace(wire).Face()

    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeShell, BRepBuilderAPI_MakeSolid
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire

    # Faces laterais da pirâmide
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCC.Core.TopTools import TopTools_ListOfShape
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.TopoDS import TopoDS_Shell, TopoDS_Solid

    # Cria faces laterais conectando a base com o ápice
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace

    edges = [
        (p1, p2),
        (p2, p3),
        (p3, p1)
    ]

    faces = []
    for edge_start, edge_end in edges:
        polygon_side = BRepBuilderAPI_MakePolygon()
        polygon_side.Add(edge_start)
        polygon_side.Add(edge_end)
        polygon_side.Add(apex)
        polygon_side.Add(edge_start)
        wire_side = polygon_side.Wire()
        face_side = BRepBuilderAPI_MakeFace(wire_side).Face()
        faces.append(face_side)

    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing
    sewing = BRepBuilderAPI_Sewing()
    sewing.Add(base_face)
    for f in faces:
        sewing.Add(f)
    sewing.Perform()
    shell = sewing.SewedShape()

    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
    solid_maker = BRepBuilderAPI_MakeSolid()
    solid_maker.Add(shell)
    return solid_maker.Solid()


def make_ellipsoid(radii):
    from OCC.Core.gp import gp_Ax2
    from OCC.Core.Geom import Geom_Ellipsoid
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeSolid, BRepBuilderAPI_MakeShell
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere

    # Sphere com escala nos 3 raios para simular elipsoide
    # OBS: pythonocc não tem elipsoide direto, vamos usar esfera e depois escalar no Blender, ou fazer malha personalizada
    # Para simplificar, geraremos esfera e anotaremos os parâmetros para pós-processamento.
    return BRepPrimAPI_MakeSphere(max(radii)).Shape()


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


# ===== 2. RENDERIZAÇÃO COM BLENDER (STL → PNG) =====
def render_stls():
    images_dir = os.path.join(OUTPUT_DIR, "images")
    os.makedirs(images_dir, exist_ok=True)

    stl_dir = os.path.join(OUTPUT_DIR, "stl")
    # Script Python do Blender para importar STL e renderizar imagens
    blender_script = f"""
import bpy
import os

output_dir = r"{os.path.abspath(OUTPUT_DIR)}"
stl_dir = os.path.join(output_dir, "stl")
images_dir = os.path.join(output_dir, "images")

for file in os.listdir(stl_dir):
    if not file.endswith(".stl"):
        continue
    filepath = os.path.join(stl_dir, file)

    # Limpa cena
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Importa STL
    bpy.ops.import_mesh.stl(filepath=filepath)
    obj = bpy.context.selected_objects[0]

    # Centraliza objeto na origem
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='MEDIAN')
    obj.location = (0, 0, 0)

    # Configura câmera
    cam = bpy.data.cameras.new("Camera")
    cam_ob = bpy.data.objects.new("Camera", cam)
    bpy.context.collection.objects.link(cam_ob)
    bpy.context.scene.camera = cam_ob
    cam_ob.location = (0, -30, 10)
    cam_ob.rotation_euler = (1.2, 0, 0)

    # Configura luz
    light_data = bpy.data.lights.new(name="light_1", type='SUN')
    light_object = bpy.data.objects.new(name="light_1", object_data=light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = (0, -10, 10)

    # Render configurações
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.filepath = os.path.join(images_dir, file.replace(".stl", ".png"))
    bpy.context.scene.render.resolution_x = {RENDER_RESOLUTION[0]}
    bpy.context.scene.render.resolution_y = {RENDER_RESOLUTION[1]}
    bpy.context.scene.render.film_transparent = True

    bpy.ops.render.render(write_still=True)
"""

    script_path = os.path.join(OUTPUT_DIR, "blender_render.py")
    with open(script_path, "w") as f:
        f.write(blender_script)

    # Ajuste o caminho do executável Blender conforme seu sistema
    blender_exec =  r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe"

    print("Renderizando STL para PNG com Blender (pode demorar)...")
    result = subprocess.run([blender_exec, "--background", "--python", script_path], capture_output=True, text=True)

    if result.returncode != 0:
        print("Erro na renderização Blender:")
        print(result.stderr)
    else:
        print("Renderização concluída com sucesso.")

    os.remove(script_path)


# ===== 3. EXECUÇÃO =====
if __name__ == "__main__":
    print("=== GERANDO DATASET 3D ===")
    generate_shapes()
    print("=== RENDERIZANDO IMAGENS ===")
    render_stls()
    print(f"Dataset pronto em: {os.path.abspath(OUTPUT_DIR)}")
