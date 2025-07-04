import bpy
import os

# === CONFIGURAÇÕES DO USUÁRIO ===
input_folder = r"C:\Venv\OpenCv\computer-vision-studies\datasets\STEP_blender\converted_stl"
output_folder = r"C:\Venv\OpenCv\computer-vision-studies\datasets\STEP_blender\rendered_images"

# === AJUSTES INICIAIS ===
def clean_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def setup_camera_and_light():
    # Remove câmera padrão se existir
    if "Camera" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Camera"])
    
    # Cria nova câmera
    bpy.ops.object.camera_add(location=(0, -2.5, 1.5), rotation=(1.0, 0, 0))
    camera = bpy.context.active_object
    camera.name = "RenderCamera"
    bpy.context.scene.camera = camera
    
    # Configura luz
    bpy.ops.object.light_add(type='SUN', location=(3, -3, 5))
    light = bpy.context.active_object
    light.data.energy = 2.0

def import_stl(filepath):
    try:
        bpy.ops.wm.stl_import(filepath=filepath)
        
        # Seleciona todos os objetos e centraliza
        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH':
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
                bpy.ops.object.location_clear()
                break
        
        return True
    except Exception as e:
        print(f"Erro ao importar STL: {filepath} -> {e}")
        return False

def setup_depth_compositor(depth_output_path):
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    # Configura o render layer para incluir o passe Z
    view_layer = bpy.context.view_layer
    view_layer.use_pass_z = True

    # Nodes
    rl = tree.nodes.new('CompositorNodeRLayers')
    normalize = tree.nodes.new('CompositorNodeNormalize')
    file_output = tree.nodes.new('CompositorNodeOutputFile')
    file_output.base_path = os.path.dirname(depth_output_path)
    file_output.file_slots[0].path = os.path.splitext(os.path.basename(depth_output_path))[0]
    file_output.format.file_format = 'OPEN_EXR'
    file_output.format.color_depth = '32'

    # Conexões
    tree.links.new(rl.outputs['Depth'], normalize.inputs[0])
    tree.links.new(normalize.outputs[0], file_output.inputs[0])

def render_model(model_path):
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    output_rgb = os.path.join(output_folder, f"{model_name}.png")
    output_depth = os.path.join(output_folder, f"{model_name}_depth.exr")

    clean_scene()
    setup_camera_and_light()

    if not import_stl(model_path):
        return

    # Configurações de renderização
    scene = bpy.context.scene
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = output_rgb
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    scene.render.film_transparent = True
    scene.view_settings.view_transform = 'Standard'

    # Verifica se há uma câmera ativa
    if not scene.camera:
        print("Erro: Nenhuma câmera definida para renderização!")
        return

    # Primeiro renderiza a imagem RGB
    print(f"Renderizando {model_name}...")
    bpy.ops.render.render(write_still=True)

    # Configura e renderiza o depth map
    setup_depth_compositor(output_depth)
    bpy.ops.render.render(write_still=True)

# === LOOP PRINCIPAL ===
def main():
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print("=== Início do processo de renderização ===")

    for file in os.listdir(input_folder):
        if file.lower().endswith('.stl'):
            model_path = os.path.join(input_folder, file)
            print(f"Processando: {model_path}")
            render_model(model_path)

    print("=== Processo finalizado ===")

if __name__ == "__main__":
    main()