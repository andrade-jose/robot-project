import bpy
import os
import sys
import math
import json
from mathutils import Vector
from random import choice

# Configurações globais
RES_X = 256
RES_Y = 256
OUTPUT_FORMAT = 'PNG'
DEPTH_FORMAT = 'OPEN_EXR'
NUM_VIEWS = 6  # máximo de vistas por modelo
CAMERA_DISTANCE = 3.0
CAMERA_HEIGHTS = [1.0, 1.5, 2.0]
CAMERA_ANGLES = [0, 60, 120, 180, 240, 300]  # graus
# Adiciona lista de cores para fundo (RGBA)
FUNDOS_CORES = [
    (1, 1, 1, 1),       # branco
    (0, 0, 0, 1),       # preto
    (0.8, 0.2, 0.2, 1), # vermelho
    (0.2, 0.8, 0.2, 1), # verde
    (0.2, 0.2, 0.8, 1), # azul
    (0.9, 0.9, 0.1, 1)  # amarelo
]

# Adiciona lista de cores para material do objeto (RGBA)
MATERIAIS_CORES = [
    (0.8, 0.8, 0.8, 1),  # Cinza claro
    (0.9, 0.5, 0.2, 1),  # Laranja
    (0.3, 0.3, 0.3, 1),  # Cinza escuro
    (0.1, 0.6, 0.9, 1),  # Azul claro
    (0.7, 0.1, 0.1, 1),  # Vermelho escuro
    (0.1, 0.7, 0.1, 1)   # Verde escuro
]
def clear_scene():
    """Limpa a cena completamente"""
    bpy.ops.wm.read_factory_settings(use_empty=True)

def setup_light():
    """Configura iluminação básica e ambiente"""
    bpy.ops.object.light_add(type='SUN', location=(5, -5, 5))
    light = bpy.context.active_object
    light.data.energy = 5.0  # Energia aumentada

    # Verifica se a cena tem um world configurado
    if bpy.context.scene.world is None:
        bpy.context.scene.world = bpy.data.worlds.new("World")

    # Configura iluminação ambiente
    bpy.context.scene.world.use_nodes = True
    world_nodes = bpy.context.scene.world.node_tree.nodes
    bg = world_nodes.get("Background")
    if bg:
        bg.inputs[1].default_value = 0.5  # Força da luz ambiente

def set_world_background(color):  # <<== Função nova para trocar fundo
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    bg = nodes.get('Background')
    if bg:
        bg.inputs[0].default_value = color  # RGBA


def import_stl(filepath):
    try:
        bpy.ops.wm.stl_import(filepath=filepath)
        objs = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
        if not objs:
            print(f"Erro: nenhum mesh importado em {filepath}")
            return False
        obj = objs[0]
        obj.location = (0, 0, 0)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='MEDIAN')
        
        # Escala adaptativa — aumenta o objeto para ficar visível
        max_dim = max(obj.dimensions)
        if max_dim < 0.1:
            scale_factor = 1.0 / max_dim  # Escala para ~1 unidade de altura
            obj.scale = (scale_factor, scale_factor, scale_factor)
            bpy.ops.object.transform_apply(scale=True)

        return True
    except Exception as e:
        print(f"Erro ao importar STL: {e}")
        return False


def apply_basic_material(context, color):
    """Aplica um material básico a todos os objetos mesh"""
    for obj in context.scene.objects:
        if obj.type == 'MESH':
            # Cria novo material
            mat = bpy.data.materials.new(name=f"Mat_{obj.name}")
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            
            # Configura shader básico
            bsdf = nodes.get("Principled BSDF")
            if bsdf:
                bsdf.inputs['Base Color'].default_value = color
                bsdf.inputs['Roughness'].default_value = 0.4
            
            # Aplica o material
            if obj.data.materials:
                obj.data.materials[0] = mat
            else:
                obj.data.materials.append(mat)

def setup_camera(angle_deg, height, distance=CAMERA_DISTANCE):
    """Configura e posiciona a câmera"""
    angle_rad = math.radians(angle_deg)
    x = distance * math.cos(angle_rad)
    y = distance * math.sin(angle_rad)
    z = height

    bpy.ops.object.camera_add(location=(x, y, z))
    cam = bpy.context.active_object

    direction = Vector((0, 0, 0)) - cam.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()

    bpy.context.scene.camera = cam
    return cam

def setup_depth_compositor(depth_output_path):
    """Configura o compositor para renderizar mapa de profundidade"""
    scene = bpy.context.scene
    
    if not scene.view_layers:
        scene.view_layers.new("ViewLayer")
    view_layer = scene.view_layers[0]  # Usa a primeira view layer
    
    view_layer.use_pass_z = True
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    rl = tree.nodes.new('CompositorNodeRLayers')
    rl.location = 0, 0

    normalize = tree.nodes.new('CompositorNodeNormalize')
    normalize.location = 200, 0

    file_output = tree.nodes.new('CompositorNodeOutputFile')
    file_output.location = 400, 0
    file_output.base_path = os.path.dirname(depth_output_path)
    file_output.file_slots[0].path = os.path.splitext(os.path.basename(depth_output_path))[0]
    file_output.format.file_format = 'OPEN_EXR'
    file_output.format.color_depth = '32'

    tree.links.new(rl.outputs['Depth'], normalize.inputs[0])
    tree.links.new(normalize.outputs[0], file_output.inputs[0])

def setup_render_engine():
    """Configura o motor de renderização Cycles"""
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE_NEXT'  # Usa Cycles como motor de renderizaçãon
    scene.cycles.samples = 128
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 12

def render_view(model_name, output_dir, angle, height, view_idx):
    """Renderiza uma vista específica"""
    scene = bpy.context.scene
    
    
    # Seleciona uma cor aleatória para fundo e material
    bg_color = choice(FUNDOS_CORES)
    mat_color = choice(MATERIAIS_CORES)

    # Aplica cor do fundo
    set_world_background(bg_color)

    # Aplica cor do material
    apply_basic_material(bpy.context, mat_color)


    # Configura view layer
    if not scene.view_layers:
        scene.view_layers.new("ViewLayer")
    view_layer = scene.view_layers[0]
    view_layer.use_pass_z = True
    
    # Configura câmera
    cam = setup_camera(angle, height)

    # Render RGB
    scene.use_nodes = False
    scene.render.image_settings.file_format = 'PNG'
    rgb_path = os.path.join(output_dir, f"{model_name}_view{view_idx:02d}.png")
    scene.render.filepath = rgb_path
    bpy.ops.render.render(write_still=True)

    # Render Depth
    depth_path = os.path.join(output_dir, f"{model_name}_view{view_idx:02d}_depth.exr")
    setup_depth_compositor(depth_path)
    scene.use_nodes = True
    scene.render.filepath = ""
    bpy.ops.render.render(write_still=True)

    # Limpeza
    bpy.data.objects.remove(cam, do_unlink=True)

    return rgb_path, depth_path, angle, height, bg_color, mat_color

def main():
    print("🔧 Iniciando renderização em lote...")

    BASE_PATH = r"C:\Venv\OpenCv\computer-vision-studies\datasets\STEP_blender"
    STL_FOLDER = os.path.join(BASE_PATH, "stl")
    RENDER_FOLDER = os.path.join(BASE_PATH, "renders")
    
    # Verificação de pastas
    if not os.path.exists(STL_FOLDER):
        print(f"❌ Pasta não encontrada: {STL_FOLDER}")
        return

    arquivos_stl = [f for f in os.listdir(STL_FOLDER) if f.lower().endswith(".stl")]
    if not arquivos_stl:
        print("⚠️ Nenhum arquivo STL encontrado.")
        return

    # Cria pasta de renders
    os.makedirs(RENDER_FOLDER, exist_ok=True)

    for stl_filename in arquivos_stl:
        nome_modelo = os.path.splitext(stl_filename)[0]
        caminho_stl = os.path.join(STL_FOLDER, stl_filename)
        pasta_saida = os.path.join(RENDER_FOLDER, nome_modelo)

        print(f"\n📦 Processando modelo: {nome_modelo}")

        try:
            # Prepara cena
            clear_scene()
            setup_light()
            setup_render_engine()

            # Importa modelo
            if not import_stl(caminho_stl):
                print(f"❌ Falha ao importar: {caminho_stl}")
                continue



            # Cria diretório de saída
            os.makedirs(pasta_saida, exist_ok=True)

            # Renderiza vistas
            views_info = []
            view_idx = 1
            for altura in CAMERA_HEIGHTS:
                for angulo in CAMERA_ANGLES:
                    if view_idx > NUM_VIEWS:
                        break
                    try:
                        rgb, depth, a, h, bg_color, mat_color = render_view(nome_modelo, pasta_saida, angulo, altura, view_idx)
                        views_info.append({
                            "view_idx": view_idx,
                            "rgb_path": os.path.relpath(rgb, RENDER_FOLDER),
                            "depth_path": os.path.relpath(depth, RENDER_FOLDER),
                            "camera_angle": a,
                            "camera_height": h,
                            "background_color": bg_color,
                            "material_color": mat_color
                        })
                        view_idx += 1
                    except Exception as e:
                        print(f"⚠️ Erro ao renderizar vista {view_idx}: {str(e)}")
                        continue

            print(f"✅ Renderização concluída: {nome_modelo}")
            
            # Salva metadados
            with open(os.path.join(pasta_saida, "views_info.json"), 'w') as f:
                json.dump(views_info, f, indent=2)

        except Exception as e:
            print(f"🚨 Erro crítico processando {nome_modelo}: {str(e)}")
            continue

    print("\n✅✅ Renderização em lote concluída com sucesso!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n🚨 Erro inesperado: {e}", file=sys.stderr)
        sys.exit(1)