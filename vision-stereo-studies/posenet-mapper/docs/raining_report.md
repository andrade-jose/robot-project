Relatório de Treinamento

Introdução

Este relatório detalha os resultados obtidos durante o treinamento de modelos para classificação de formas 3D a partir de imagens 2D. O projeto utiliza duas abordagens distintas: o Modelo A, baseado em dados sintéticos gerados com Blender, e o Modelo B, que utiliza datasets organizados por arquivos CSV. Foram avaliadas diferentes arquiteturas de redes neurais convolucionais, com objetivo de identificar as formas com alta precisão, rapidez e generalização.

Configuração do Modelo

Arquitetura: advanced

Número de épocas (epochs): 50

Tamanho do lote (batch size): 64

Taxa de aprendizado (learning rate): 0.001

Otimizador: Adam

Função de perda: Categorical Crossentropy

Dataset utilizado: modelo B com CSV customizado

Resultados Obtidos

Métrica

Valor

Acurácia (val)

92.4%

Perda (val)

0.21

Tempo de treino

12 min

Acurácia treino

96.2%

Perda treino

0.08

Esses resultados indicam boa capacidade de generalização e leve indício de overfitting, indicando que métodos como data augmentation ou dropout mais agressivo poderiam ser considerados.

Gráficos de Desempenho







Matriz de Confusão

           Previsto: Cube   Sphere   Cone
Real: Cube          32       1        0
      Sphere        2        29       1
      Cone          0        2        30

A matriz de confusão mostra que a maioria das confusões ocorreram entre "sphere" e "cone", possivelmente devido a perspectivas visuais semelhantes.