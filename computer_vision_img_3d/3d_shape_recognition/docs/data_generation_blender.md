Geração de Dados Sintéticos com Blender

Requisitos

Blender versão 2.9 ou superior instalado

Python embutido habilitado

A execução é feita via terminal (modo headless):

blender --background --python src/data_generation/blender_generator.py

Configurações Personalizáveis

n_samples: Quantidade de imagens por classe

classes: Lista de formas a gerar (ex: cube, sphere, cone)

image_size: Resolução final das imagens

save_path: Diretório de destino (por padrão: data/raw/)

A geração sintética permite controlar totalmente o ambiente de treino, com vantagens em comparação a datasets reais desbalanceados