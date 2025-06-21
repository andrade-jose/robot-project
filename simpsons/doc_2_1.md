# Documentação do Sistema de Classificação de Personagens dos Simpsons

## Visão Geral
Este sistema implementa uma rede neural convolucional (CNN) para classificação de imagens dos 10 personagens principais dos Simpsons. O modelo é capaz de aprender características visuais distintivas de cada personagem e fazer previsões com base em novas imagens de entrada.

## Requisitos
- Python 3.7+
- Bibliotecas: TensorFlow/Keras, OpenCV, Caer, Numpy, Matplotlib

## Fluxo de Processamento

### 1. Pré-processamento dos Dados
```python
# Carrega e prepara o dataset
char_dict = {}
for char in os.listdir(char_path):
    char_dict[char] = len(os.listdir(os.path.join(char_path, char)))
```

- **Seleção dos Personagens**: Seleciona automaticamente os 10 personagens com mais imagens disponíveis
- **Pré-processamento**: Redimensiona todas as imagens para 80x80 pixels em escala de cinza
- **Normalização**: Converte valores de pixel para intervalo [0,1]
- **Divisão dos Dados**: 80% para treino e 20% para validação

### 2. Arquitetura da Rede Neural
A CNN possui três blocos principais:

1. **Bloco Convolucional 1**:
   - 2 camadas Conv2D (32 filtros)
   - BatchNormalization
   - MaxPooling (redução para 40x40)
   - Dropout (20%)

2. **Bloco Convolucional 2**:
   - 2 camadas Conv2D (64 filtros)
   - BatchNormalization
   - MaxPooling (redução para 20x20)
   - Dropout (30%)

3. **Bloco Convolucional 3**:
   - 2 camadas Conv2D (128 filtros)
   - BatchNormalization
   - MaxPooling (redução para 10x10)
   - Dropout (40%)

4. **Camadas Densas**:
   - Flatten (achata os features)
   - Dense (512 neurônios)
   - Camada de saída com ativação softmax (10 classes)

### 3. Treinamento
```python
optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(optimizer=optimizer, loss=CategoricalCrossentropy(), metrics=['accuracy'])
```

- **Otimizador**: SGD com momentum (Nesterov)
- **Learning Rate Schedule**: Reduz a taxa pela metade a cada 10 épocas
- **Early Stopping**: Interrompe se não houver melhoria em 15 épocas
- **Data Augmentation**: Gera variações das imagens durante o treino

### 4. Predição
```python
def prepare(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, IMG_SIZE)
    img = caer.reshape(img, IMG_SIZE, 1)
    img = img / 255.0
    return np.expand_dims(img, axis=0)
```

A função `prepare()` realiza:
1. Conversão para escala de cinza
2. Redimensionamento para 80x80 pixels
3. Normalização dos valores dos pixels
4. Adequação ao formato esperado pelo modelo (adiciona dimensões de batch)

## Como Usar

1. **Treinamento**:
```python
python simpsons_classifier.py --mode train --dataset path/to/dataset
```

2. **Predição**:
```python
python simpsons_classifier.py --mode predict --image path/to/image.jpg
```

## Estratégias de Melhoria

1. **Dados**:
   - Balanceamento das classes
   - Coleta de mais imagens variadas

2. **Modelo**:
   - Testar arquiteturas pré-treinadas (Transfer Learning)
   - Ajuste fino de hiperparâmetros

3. **Produção**:
   - Conversão para TensorFlow Lite (uso em dispositivos móveis)
   - Criação de API REST para classificação

## Exemplo de Saída
Ao processar uma imagem do Homer Simpson:
```
Personagem previsto: homer_simpson [98.7% de confiança]
```

## Observações
- O modelo foi otimizado para personagens principais dos Simpsons
- Pode requerer ajustes para outros personagens secundários
- Desempenho depende da qualidade e variedade das imagens de treino 