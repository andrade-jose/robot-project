Projeto: Geração e Renderização de Modelos 3D com Metadados
Este projeto automatiza a geração de formas geométricas 3D em arquivos STEP, sua conversão para STL com o FreeCAD, renderização com o Blender, e organização dos dados em formato CSV com metadados completos.

📂 Estrutura de Diretórios Esperada
bash
Copiar
Editar
datasets/
├── STEP_blender/
│   ├── step/          # Arquivos .step gerados
│   ├── stl/           # Arquivos .stl convertidos
│   ├── renders/
│   │   ├── model_x/
│   │   │   ├── view01.png
│   │   │   ├── view01_depth.exr
│   │   │   └── views_info.json
│   └── metadata.json  # Metadados dos modelos gerados
🧩 Requisitos
Python

OCC (PythonOCC)

FreeCAD (via bindings e execução local)

Blender (com execução de scripts via CLI)

bpy, json, csv, mathutils, math, os, random, etc.

🔧 Execução dos Módulos
1. generate_step.py
Gera 250 formas para cada tipo geométrico (cube, sphere, etc.) em arquivos STEP, salvando parâmetros e caminhos em metadata.json.

bash
Copiar
Editar
python generate_step.py
2. convert_step_to_stl.py
Converte os arquivos .step gerados para .stl usando o FreeCAD.

bash
Copiar
Editar
python convert_step_to_stl.py
3. blender_render.py
Renderiza 6 vistas (RGB e profundidade) de cada modelo .stl com diferentes ângulos, alturas e variações de fundo e material.

bash
Copiar
Editar
blender --background --python blender_render.py
4. generate_metadata_csv.py
Gera um CSV unificado com os metadados de cada render, consolidando os arquivos JSON de cada modelo.

bash
Copiar
Editar
python generate_metadata_csv.py
📊 Saída Final
O CSV contém:

model_name, shape_type, view_idx

Caminhos relativos para rgb_path e depth_path

Ângulo e altura da câmera

Cor do fundo e do material

Caminho do STL associado

📌 Observações
O sistema usa cores aleatórias para fundo e material a cada renderização.

É possível configurar a quantidade de vistas, resolução, distância da câmera, etc.

O pipeline pode ser adaptado para gerar datasets sintéticos para aprendizado profundo (DL/ML).

