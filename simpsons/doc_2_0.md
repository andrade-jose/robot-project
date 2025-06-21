
# Documentação do Código – Classificação de Personagens dos Simpsons com CNN

---

## 1. Objetivo

Construir, treinar e validar um modelo de rede neural convolucional (CNN) para classificar imagens de personagens da série Simpsons. O modelo reconhece 10 personagens diferentes a partir de imagens em escala de cinza, redimensionadas para 80x80 pixels.

---

## 2. Estrutura Geral do Código

1. **Importação de bibliotecas**
2. **Definição de parâmetros**
3. **Preparação dos dados**
4. **Definição da arquitetura do modelo**
5. **Configuração do otimizador e compilação**
6. **Data augmentation (aumento de dados)**
7. **Configuração do treinamento (callbacks, agendamento da taxa de aprendizado)**
8. **Treinamento do modelo**
9. **Função para preparar imagens para predição**
10. **Teste do modelo com uma imagem de exemplo**

---

## 3. Detalhamento das Partes

### 3.1 Importação de Bibliotecas

* `os`, `math`, `gc`: manipulação de arquivos, operações matemáticas, limpeza de memória.
* `numpy`, `cv2 (OpenCV)`, `matplotlib.pyplot`: manipulação e visualização de imagens.
* `caer` e `canaro`: bibliotecas auxiliares para processamento de imagens e criação de datasets.
* Módulos do TensorFlow/Keras para construção, treinamento e avaliação de redes neurais.

---

### 3.2 Parâmetros Principais

* `IMG_SIZE = (80, 80)`: tamanho para redimensionar as imagens.
* `CHANNELS = 1`: imagens são em escala de cinza.
* `OUTPUT_DIM = 10`: quantidade de classes (personagens) para classificação.
* `BATCH_SIZE = 128`: quantidade de amostras processadas por iteração.
* `EPOCHS = 50`: número máximo de ciclos completos de treinamento.

---

### 3.3 Preparação dos Dados

* Lê o diretório com as imagens dos personagens.
* Conta o número de imagens por personagem.
* Ordena os personagens pelo número de imagens em ordem decrescente.
* Seleciona os 10 personagens com mais imagens para o treinamento.
* Usa funções do `caer` para:

  * Pré-processar imagens (redimensionar, converter canais).
  * Separar imagens e rótulos.
  * Criar conjuntos de treino e validação (80% treino, 20% validação).
* Remove variáveis pesadas para liberar memória.

---

### 3.4 Definição do Modelo (CNN)

Modelo sequencial com as seguintes camadas:

* **Entrada**: imagens 80x80x1.
* **Bloco 1**: 2 camadas Conv2D (32 filtros) + Batch Normalization + MaxPooling + Dropout.
* **Bloco 2**: 2 camadas Conv2D (64 filtros) + Batch Normalization + MaxPooling + Dropout.
* **Bloco 3**: 2 camadas Conv2D (128 filtros) + Batch Normalization + MaxPooling + Dropout.
* **Camadas densas**: Flatten + Dense 512 neurônios + Batch Normalization + Dropout.
* **Saída**: Dense com 10 neurônios e softmax para classificação multi-classe.

---

### 3.5 Otimizador e Compilação

* Otimizador SGD com:

  * learning rate inicial 0.01,
  * momentum 0.9,
  * Nesterov ativado,
  * regularização L2 (`weight_decay`).
* Loss: `CategoricalCrossentropy` (compatível com one-hot encoding).
* Métrica de avaliação: acurácia.

---

### 3.6 Data Augmentation (Aumento de Dados)

* Para o treino, são aplicadas transformações aleatórias para melhorar a robustez do modelo:

  * rotações, deslocamentos, cisalhamento, zoom, flips horizontais.
  * normalização dos pixels para valores entre 0 e 1.
* Para validação, somente normalização é aplicada (sem aumentos).

---

### 3.7 Callbacks e Agendamento de Learning Rate

* `LearningRateScheduler`: reduz a taxa de aprendizado pela metade a cada 10 épocas.
* `EarlyStopping`: interrompe o treino se a validação não melhorar por 15 épocas, restaurando os melhores pesos.
* `ModelCheckpoint`: salva o modelo com melhor acurácia na validação.

---

### 3.8 Treinamento do Modelo

* Usa os geradores de dados para alimentar o modelo.
* Configura número de passos por época e validação para garantir que sempre haja pelo menos um passo.
* Treina por até 50 épocas (com possibilidade de parada antecipada).

---

### 3.9 Função para Preparar Imagens para Predição

* Converte imagens coloridas para escala de cinza.
* Redimensiona para 80x80.
* Ajusta shape para (1, 80, 80, 1).
* Normaliza pixels para \[0,1].

---

### 3.10 Teste com Imagem Externa

* Carrega uma imagem exemplo.
* Exibe a imagem usando Matplotlib.
* Executa a predição com o modelo treinado.
* Exibe o nome do personagem previsto.

---

## 4. Observações e Possíveis Melhorias

* **Avaliação detalhada:** Gráficos de acurácia e perda durante treino e validação podem ajudar a entender o aprendizado.
* **Transfer learning:** usar modelos pré-treinados pode melhorar resultados com menos dados.
* **Experimentar arquiteturas diferentes:** mais camadas, ResNet, MobileNet, etc.
* **Ajuste fino de hiperparâmetros:** taxa de aprendizado, batch size, dropout, etc.
* **Melhor tratamento do dataset:** balanceamento, remoção de ruído, etc.

---

Se quiser, posso ajudar a gerar essa documentação em Markdown, PDF ou outro formato. Quer?
