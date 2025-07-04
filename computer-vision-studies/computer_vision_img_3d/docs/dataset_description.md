Descrição dos Conjuntos de Dados

Modelo A (gerado com Blender)

Utiliza imagens 2D renderizadas a partir de modelos 3D criados no Blender

Classes consideradas: cube, sphere, cone, cylinder, pyramid, torus

Cada classe possui aproximadamente 500 amostras

Imagens organizadas em pastas, uma para cada classe

Resolução padrão: 224x224 px

Esse modelo é ideal para gerar dados personalizados, controlar iluminação, textura, rotação e escala.

Modelo B (definido via CSV)

As imagens estão referenciadas por caminho no disco e rótulo correspondente em um arquivo .csv

filepath,label
/data/image1.png,cube
/data/image2.png,sphere

O formato é adequado para grandes datasets, permite manipulação com pandas e é ideal para pipelines automatizadas

Suporte a dados externos como Kaggle, Google Drive ou bancos de imagens públicos

