import os
import sys

# Apenas se necessário
FREECAD_PATH = r"C:\Program Files\FreeCAD 1.0\bin"
if FREECAD_PATH not in sys.path:
    sys.path.append(FREECAD_PATH)

import FreeCAD
import Part
import Mesh

input_folder = r"C:\Venv\OpenCv\computer-vision-studies\datasets\STEP_blender\generated_step"
output_folder = r"C:\Venv\OpenCv\computer-vision-studies\datasets\STEP_blender\converted_stl"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(".step"):
        step_path = os.path.join(input_folder, filename)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_folder, base_name + ".stl")

        try:
            doc = FreeCAD.newDocument()
            
            # Substitui ImportGui por Part.read
            shape = Part.read(step_path)
            part_obj = doc.addObject("Part::Feature", "Imported")
            part_obj.Shape = shape

            Mesh.export([part_obj], output_path)
            FreeCAD.closeDocument(doc.Name)
            print(f"✔️ Convertido: {filename} → {output_path}")
        except Exception as e:
            print(f"❌ Falha ao converter {filename}: {e}")
