import bpy
import math
import random
import os

# Pasta onde salvar
root_dir = r'C:\Venv\OpenCv\opencv_live_course\Datashet'
formas = {
    "cubo": lambda loc: bpy.ops.mesh.primitive_cube_add(size=1, location=loc),
    "esfera": lambda loc: bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=loc),
    "cone": lambda loc: bpy.ops.mesh.primitive_cone_add(radius1=1, depth=2, location=loc),
    "cilindro": lambda loc: bpy.ops.mesh.primitive_cylinder_add(radius=1, depth=2, location=loc),
    "paralelepipido": lambda loc: bpy.ops.mesh.primitive_cube_add(scale=(1, 0.5, 2), location=loc),
    "piramide": lambda loc: bpy.ops.mesh.primitive_cone_add(vertices=4, radius1=1, depth=2, location=loc),
}

# Limpar cena
bpy.ops.wm.read_factory_settings(use_empty=True)

# Adicionar câmera
cam_data = bpy.data.cameras.new(name="Camera")
cam_obj = bpy.data.objects.new("Camera", cam_data)
bpy.context.collection.objects.link(cam_obj)
cam_obj.location = (0, -5, 2)
cam_obj.rotation_euler = (math.radians(75), 0, 0)
bpy.context.scene.camera = cam_obj

# Adicionar luz
light_data = bpy.data.lights.new(name="Luz", type='SUN')
light_obj = bpy.data.objects.new("Luz", light_data)
bpy.context.collection.objects.link(light_obj)
light_obj.location = (0, 0, 10)

# Loop para criar imagens
for forma, criador in formas.items():
    output_dir = os.path.join(root_dir, forma)
    os.makedirs(output_dir, exist_ok=True)

    for i in range(100):  # Gera 100 imagens por forma
        # Limpa objetos (menos câmera e luz)
        for obj in bpy.data.objects:
            if obj.type not in {'CAMERA', 'LIGHT'}:
                bpy.data.objects.remove(obj, do_unlink=True)

        # Cria objeto usando a função lambda
        criador(loc=(0, 0, 0))  # Agora funciona!
        objeto = bpy.context.view_layer.objects.active

        # Rotação aleatória
        objeto.rotation_euler = (
            random.uniform(0, 2*math.pi),
            random.uniform(0, 2*math.pi),
            random.uniform(0, 2*math.pi)
        )

        # Cor aleatória
        mat = bpy.data.materials.new(name="Material")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (
            random.random(), random.random(), random.random(), 1
        )
        objeto.data.materials.append(mat)

        # Render
        bpy.context.scene.render.filepath = os.path.join(output_dir, f"{forma}_{i:03d}.png")
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.ops.render.render(write_still=True)

print("✅ Imagens geradas com sucesso.")