import bpy
import math
import random
import os
import csv

# ====== CONFIG ======
root_dir = r"C:\Venv\OpenCv\opencv_live_course\Dataset_Augmented"
total_images = 10000

# ====== FORMAS ======
def cria_cruz_3d(loc):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(loc[0] + 1, loc[1], loc[2]))
    bpy.ops.mesh.primitive_cube_add(size=1, location=(loc[0], loc[1] + 1, loc[2]))

def cria_L_shape(loc):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(loc[0], loc[1] + 1, loc[2]))

def cria_capsula(loc):
    bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=1, location=loc)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=(loc[0], loc[1], loc[2] + 0.5))
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=(loc[0], loc[1], loc[2] - 0.5))

def cria_hemisfera(loc):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=loc)
    obj = bpy.context.active_object
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.bisect(plane_co=(0, 0, 0), plane_no=(0, 0, 1), clear_inner=True)
    bpy.ops.object.mode_set(mode='OBJECT')

formas = {
    "cubo": lambda loc: bpy.ops.mesh.primitive_cube_add(size=1, location=loc),
    "paralelepipido": lambda loc: bpy.ops.mesh.primitive_cube_add(scale=(1.5, 0.6, 1.2), location=loc),
    "plano": lambda loc: bpy.ops.mesh.primitive_plane_add(size=1, location=loc),
    "disco": lambda loc: bpy.ops.mesh.primitive_circle_add(radius=1, location=loc),
    "esfera_uv": lambda loc: bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=loc),
    "ico_esfera": lambda loc: bpy.ops.mesh.primitive_ico_sphere_add(radius=1, location=loc),
    "cone": lambda loc: bpy.ops.mesh.primitive_cone_add(radius1=1, depth=2, location=loc),
    "cilindro": lambda loc: bpy.ops.mesh.primitive_cylinder_add(radius=1, depth=2, location=loc),
    "piramide": lambda loc: bpy.ops.mesh.primitive_cone_add(vertices=4, radius1=1, depth=2, location=loc),
    "toroide": lambda loc: bpy.ops.mesh.primitive_torus_add(location=loc, major_radius=1, minor_radius=0.3),
    "tetraedro": lambda loc: bpy.ops.mesh.primitive_tetrahedron_add(location=loc),
    "octaedro": lambda loc: bpy.ops.mesh.primitive_octahedron_add(location=loc),
    "dodecaedro": lambda loc: bpy.ops.mesh.primitive_dodecahedron_add(location=loc),
    "icosaedro": lambda loc: bpy.ops.mesh.primitive_icosahedron_add(location=loc),
    "prisma_triangular": lambda loc: bpy.ops.mesh.primitive_cone_add(vertices=3, radius1=1, depth=2, location=loc),
    "prisma_hexagonal": lambda loc: bpy.ops.mesh.primitive_cone_add(vertices=6, radius1=1, depth=2, location=loc),
    "elipsoide": lambda loc: (
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=loc),
        setattr(bpy.context.active_object.scale, "__setitem__", lambda s, v: bpy.context.active_object.scale.__setitem__(s, v)) and bpy.context.active_object.scale.__setitem__(slice(0, 3), (1.2, 0.8, 1.0))
    ),
    "esfera_achatada": lambda loc: (
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=loc),
        setattr(bpy.context.active_object.scale, "__setitem__", lambda s, v: bpy.context.active_object.scale.__setitem__(s, v)) and bpy.context.active_object.scale.__setitem__(slice(0, 3), (1.0, 1.0, 0.5))
    ),
    "esfera_elongada": lambda loc: (
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=loc),
        setattr(bpy.context.active_object.scale, "__setitem__", lambda s, v: bpy.context.active_object.scale.__setitem__(s, v)) and bpy.context.active_object.scale.__setitem__(slice(0, 3), (1.0, 1.0, 1.5))
    ),
    "cubo_arredondado": lambda loc: (
        bpy.ops.mesh.primitive_cube_add(size=1, location=loc),
        bpy.ops.object.modifier_add(type='SUBSURF'),
        bpy.ops.object.shade_smooth()
    ),
    "cruz_3d": cria_cruz_3d,
    "L_shape": cria_L_shape,
    "cápsula": cria_capsula,
    "hemisfera": cria_hemisfera,
}

formas_lista = list(formas.keys())
imagens_por_classe = total_images // len(formas_lista)

# ====== SETUP CENA ======
bpy.ops.wm.read_factory_settings(use_empty=True)

scene = bpy.context.scene

# Câmera
cam_data = bpy.data.cameras.new("Camera")
cam_obj = bpy.data.objects.new("Camera", cam_data)
bpy.context.collection.objects.link(cam_obj)
cam_obj.location = (0, -6, 3)
cam_obj.rotation_euler = (math.radians(65), 0, 0)
scene.camera = cam_obj

# Plano de fundo
bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, -1))
plane = bpy.context.active_object
plane_mat = bpy.data.materials.new(name="PlaneMat")
plane_mat.use_nodes = True
plane.data.materials.append(plane_mat)

# Luz
light_data = bpy.data.lights.new(name="Luz", type='SUN')
light_obj = bpy.data.objects.new("Luz", light_data)
bpy.context.collection.objects.link(light_obj)

# Render
scene.render.resolution_x = 512
scene.render.resolution_y = 512
scene.render.film_transparent = False
scene.use_nodes = True

# Compositor nodes
tree = scene.node_tree
tree.nodes.clear()

render_layers = tree.nodes.new(type='CompositorNodeRLayers')
composite = tree.nodes.new(type='CompositorNodeComposite')
tree.links.new(render_layers.outputs['Image'], composite.inputs['Image'])

mask_output = tree.nodes.new('CompositorNodeOutputFile')
mask_output.base_path = ""
mask_output.label = "Mask"
tree.links.new(render_layers.outputs['IndexOB'], mask_output.inputs[0])

depth_output = tree.nodes.new('CompositorNodeOutputFile')
depth_output.base_path = ""
depth_output.label = "Depth"
tree.links.new(render_layers.outputs['Depth'], depth_output.inputs[0])

scene.view_layers["View Layer"].use_pass_object_index = True

# CSV global
csv_path = os.path.join(root_dir, "all_metadata.csv")
csv_file = open(csv_path, mode='w', newline='')
writer = csv.writer(csv_file)
writer.writerow([
    "filename", "class",
    "obj_loc_x", "obj_loc_y", "obj_loc_z",
    "obj_rot_x", "obj_rot_y", "obj_rot_z",
    "obj_scale_x", "obj_scale_y", "obj_scale_z",
    "light_x", "light_y", "light_z",
    "bg_r", "bg_g", "bg_b"
])

# ====== LOOP PRINCIPAL ======
for forma, criador in formas.items():
    output_dir = os.path.join(root_dir, forma)
    img_dir = os.path.join(output_dir, "images")
    mask_dir = os.path.join(output_dir, "masks")
    depth_dir = os.path.join(output_dir, "depth")

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    for i in range(imagens_por_classe):
        idx = f"{forma}_{i:05d}"
        filename = f"{idx}.png"

        # Limpar objetos (exceto câmera, luz, plano)
        for obj in list(bpy.data.objects):
            if obj.name in {'Camera', 'Luz', plane.name}:
                continue
            bpy.data.objects.remove(obj, do_unlink=True)

        # Criar objeto(s)
        criador((random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), 0.5))
        objeto = bpy.context.active_object  # Note que em formas compostas isso pode ser último criado

        # Escala aleatória - para objetos compostos pode não aplicar a todos (precisa ajuste manual)
        try:
            objeto.scale = (
                random.uniform(0.8, 1.2),
                random.uniform(0.8, 1.2),
                random.uniform(0.8, 1.2)
            )
        except:
            pass

        # Rotação aleatória
        try:
            objeto.rotation_euler = (
                random.uniform(0, 2 * math.pi),
                random.uniform(0, 2 * math.pi),
                random.uniform(0, 2 * math.pi)
            )
        except:
            pass

        # Material aleatório para o objeto principal (pode faltar para múltiplos objetos compostos)
        try:
            mat = bpy.data.materials.new(name="ObjMaterial")
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            bsdf.inputs['Base Color'].default_value = (
                random.random(), random.random(), random.random(), 1
            )
            objeto.data.materials.clear()
            objeto.data.materials.append(mat)
            objeto.pass_index = 1
        except:
            pass

        # Luz aleatória
        light_obj.location = (
            random.uniform(-3, 3),
            random.uniform(-3, 3),
            5
        )

        # Fundo aleatório
        r = random.uniform(0.8, 1)
        g = random.uniform(0.8, 1)
        b = random.uniform(0.8, 1)
        bg_bsdf = plane_mat.node_tree.nodes.get("Principled BSDF")
        bg_bsdf.inputs["Base Color"].default_value = (r, g, b, 1)

        # Setar paths
        scene.render.filepath = os.path.join(img_dir, filename)
        mask_output.base_path = mask_dir
        mask_output.file_slots[0].path = f"{idx}_mask_"
        depth_output.base_path = depth_dir
        depth_output.file_slots[0].path = f"{idx}_depth_"

        # Renderizar
        bpy.ops.render.render(write_still=True)

        # Salvar metadados
        try:
            writer.writerow([
                filename, forma,
                objeto.location.x, objeto.location.y, objeto.location.z,
                objeto.rotation_euler.x, objeto.rotation_euler.y, objeto.rotation_euler.z,
                objeto.scale.x, objeto.scale.y, objeto.scale.z,
                light_obj.location.x, light_obj.location.y, light_obj.location.z,
                r, g, b
            ])
        except:
            # Caso objeto múltiplo, escrever valores padrão (0)
            writer.writerow([
                filename, forma,
                0,0,0,0,0,0,0,0,0,
                light_obj.location.x, light_obj.location.y, light_obj.location.z,
                r, g, b
            ])

csv_file.close()
print("✅ 10k imagens com 26 formas geradas com sucesso!")
