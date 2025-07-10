
# 🧠 Multiview CNN 3D Shape Recognition

Este projeto implementa uma arquitetura de rede neural profunda baseada em visão multiview para reconhecimento de formas 3D a partir de imagens RGB, mapas de profundidade e variáveis auxiliares (ex: cor do fundo e material). Ele inclui pipeline completo de pré-processamento, augmentação, treinamento, validação, callbacks e salvamento de modelos.

---

## 📁 Estrutura do Projeto

```
.
├── architectures/
│   └── advanced_cnn.py      # Arquitetura Multiview CNN com SE Blocks e LSTM
├── config/
│   └── config_advanced.py   # Configurações gerais do projeto
├── data/
│   ├── loading_csv.py       # DataLoader compatível com CSV + EXR
│   └── preprocessing.py     # Augmentações com imgaug
├── training/
│   └── callbacks.py         # Callbacks Keras para treino avançado
├── train_advanced.py        # Script principal de treinamento
└── README.md
```

---

## 🧠 Modelo Multiview

A arquitetura principal (`Multiview_CNN`) é composta por:

* **Módulos Residual SE Blocks** para extração de características por vista (RGB).
* **Processamento 1D de mapas de profundidade**.
* **Concatenação com variáveis auxiliares** (opcional).
* **Bidirectional LSTM** para modelar dependência entre vistas.
* **Classificação final por Dense + softmax**.

### Entradas:

* `rgb_input`: Tensor (batch, 6, H, W, 3)
* `depth_input`: Tensor (batch, 6, H, W, 1)
* `aux_input`: Tensor (batch, 6, 6) *(opcional)*

---

## ⚙️ Configuração

As configurações gerais são definidas em `config/config_advanced.py`, incluindo:

* Tamanho de imagem: `IMG_SIZE = (125, 125)`
* Número de classes: `NUM_CLASSES = 7`
* Diretórios de dados, logs e modelos
* Parâmetros de augmentação e divisão de dados
* Uso de GPU e mixed precision (automático)

---

## 🧪 Dataset

* Espera-se que as imagens estejam organizadas com caminhos registrados em um arquivo `dataset.csv`.

* Cada modelo 3D deve possuir **6 vistas** com colunas:

  * `model_name`, `view_idx`, `rgb_path`, `depth_path`, `shape_type`, `background_color`, `material_color`

* Os **depth maps** devem estar no formato **.EXR** com canal `'R'`.

---

## 🧼 Augmentações

Em `data/preprocessing.py`, as imagens RGB e mapas de profundidade são aumentadas simultaneamente usando:

* Flips, rotações, shear, escala
* Ruído gaussiano, contraste, brilho

---

## 🚀 Treinamento

### Comando:

```bash
python train_advanced.py --optimizer adam --lr 0.001 --epochs 50 --batch_size 16
```

### Argumentos principais:

| Argumento           | Descrição                                             |
| ------------------- | ----------------------------------------------------- |
| `--optimizer`       | Escolha entre: `adam`, `adamw`, `sgd`, `rmsprop`      |
| `--lr`              | Taxa de aprendizado inicial                           |
| `--epochs`          | Número de épocas de treino                            |
| `--batch_size`      | Tamanho do lote                                       |
| `--use_pretrained`  | Usa backbone pré-treinado (não habilitado atualmente) |
| `--freeze_backbone` | Congela o backbone (caso `use_pretrained`)            |

---

## 🧩 Callbacks

Fornecidos por `training/callbacks.py`:

* `EarlyStopping`
* `ModelCheckpoint`
* `ReduceLROnPlateau`
* `TensorBoard`
* `CSVLogger`
* Agendador de LR com decaimento exponencial

---

## 📈 Logs e Modelos

* Modelos são salvos em `models/trained/`
* Logs e históricos em `models/logs/`
* Históricos em `.csv` e `.npy`

---

## 📊 Métricas

Durante o treino são monitoradas:

* `accuracy`
* `top3_accuracy`
* `top5_accuracy`
* `val_loss`

---

## ✅ Requisitos

* Python 3.8+
* TensorFlow 2.8+
* OpenCV
* OpenEXR e Imath
* imgaug
* pandas, numpy

Instale com:

```bash
pip install -r requirements.txt
```

Exemplo de `requirements.txt`:

```txt
tensorflow>=2.8
opencv-python
imgaug
openexr
imath
pandas
numpy
```

---

## 🧪 Avaliação

Após o treino, o script executa avaliação no conjunto de teste com logging detalhado e salvamento automático do melhor modelo.


