basic

Três camadas convolucionais com ReLU + MaxPooling

Uma camada totalmente conectada (Dense) com softmax de saída

Arquitetura simples, recomendada para protótipos ou dispositivos com baixo poder computacional


advanced

Base: EfficientNetB0 com customizações

Inclui SE blocks (Squeeze-and-Excitation) para ajuste dinâmico de canais

Batch Normalization e Dropout para evitar overfitting

Ideal para aplicações em GPUs, mantendo bom balanço entre desempenho e custo computacional


hybrid

Combina as saídas dos modelos basic e advanced utilizando ensemble

A decisão final é tomada por voto majoritário ou média das probabilidades

Permite robustez extra em cenários com ruído ou incerteza