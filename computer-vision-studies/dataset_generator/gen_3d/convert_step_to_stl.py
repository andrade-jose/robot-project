import sys
import os

# ➕ Caminhos para as bibliotecas do FreeCAD
FREECAD_PATH = r"C:\Program Files\FreeCAD 1.0\bin"
FREECAD_LIB = r"C:\Program Files\FreeCAD 1.0\lib"

sys.path.append(FREECAD_PATH)
sys.path.append(FREECAD_LIB)
os.environ["FREECAD_LIB"] = FREECAD_LIB

# ➕ Importações do FreeCAD
import FreeCAD
import Part
import Mesh

def converter_step_para_stl(caminho_step, caminho_stl, precisao_malha=0.1):
    """Converte um arquivo STEP para STL usando o FreeCAD"""
    try:
        doc = FreeCAD.newDocument("Conversao")
        forma = Part.read(caminho_step)
        objeto_parte = doc.addObject("Part::Feature", "ParteImportada")
        objeto_parte.Shape = forma

        diretorio_saida = os.path.dirname(caminho_stl)
        if not os.path.exists(diretorio_saida):
            os.makedirs(diretorio_saida)

        Mesh.export([objeto_parte], caminho_stl)
        print(f"✓ {os.path.basename(caminho_step)} → Conversão bem-sucedida")

        FreeCAD.closeDocument(doc.Name)
        return True

    except Exception as erro:
        print(f"✗ {os.path.basename(caminho_step)} → Erro na conversão: {erro}", file=sys.stderr)
        return False

if __name__ == "__main__":
    pasta_step = r"C:\Venv\OpenCv\computer-vision-studies\datasets\STEP_blender\step"
    pasta_stl = r"C:\Venv\OpenCv\computer-vision-studies\datasets\STEP_blender\stl"
    precisao_malha = 0.1

    if not os.path.exists(pasta_step):
        print(f"Erro: Pasta STEP não encontrada: {pasta_step}", file=sys.stderr)
        sys.exit(1)

    arquivos_step = [f for f in os.listdir(pasta_step) if f.lower().endswith((".step", ".stp"))]
    if not arquivos_step:
        print("Nenhum arquivo STEP encontrado na pasta.")
        sys.exit(0)

    print(f"⏳ Convertendo {len(arquivos_step)} arquivos STEP para STL...")

    for nome_arquivo in arquivos_step:
        caminho_step = os.path.join(pasta_step, nome_arquivo)
        nome_stl = os.path.splitext(nome_arquivo)[0] + ".stl"
        caminho_stl = os.path.join(pasta_stl, nome_stl)

        converter_step_para_stl(caminho_step, caminho_stl, precisao_malha)

    print("✅ Conversão em lote concluída com sucesso!")
