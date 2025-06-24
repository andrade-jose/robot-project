
```markdown
# **Sistema de Reconhecimento de Formas 3D**

Este projeto implementa um sistema completo de reconhecimento de formas geométricas 3D com **visão computacional** e **deep learning**. São disponibiladas **duas versões distintas** de pipeline de dados e modelo:

- **Modelo A**: baseado em imagens geradas com Blender, organizadas em pastas por classe.
- **Modelo B**: baseado em datasets definidos via CSV, com suporte a múltiplas arquiteturas e datasets externos.

---

## 📦 Estrutura do Projeto
```

3d\_shape\_recognition/
├── bases/                       # 📂 Arquivos de origem e referência
├── data/
│   ├── raw/                     # ⚠️ Dados brutos
│   ├── processed/               # ⚠️ Dados pré-processados
│   └── samples/                 # 📸 Amostras de teste
├── docs/                        # 📚 Documentação
├── models/
│   ├── logs/                    # 📊 Logs de treinamento (TensorBoard)
│   ├── train/                   # 📈 Resultados do treinamento
│   ├── validation/              # ✅ Validações do modelo
│   └── trained/                 # 🧠 Modelos salvos (.h5)
├── src/
│   ├── bases/                   # 🛠️ Arquivos de origem comuns
│   ├── data\_generation/         # 🌀 Geração de dados no Blender (modelo A)
│   ├── data\_processing\_a/       # 🔄 Pré-processamento para o modelo A
│   ├── data\_processing\_b/       # 🔄 Pré-processamento para o modelo B (via CSV)
│   ├── inference/               # \[OBSOLETO] (inferência real-time desativada)
│   ├── training\_model\_a/        # 🧠 Treinamento versão A (dados do Blender)
│   ├── training\_model\_b/        # 🧠 Treinamento versão B (flexível e avançado)
│   └── utils/                   # 🧩 Funções auxiliares
├── tests/                       # 🧪 Testes unitários
├── requirements.txt             # 📜 Dependências do Python
└── README.md                    # 📘 Este arquivo

````

## 🚀 Como Usar

### 1️⃣ Instalação
```bash
git clone https://github.com/seu-usuario/3d-shape-recognition.git
cd 3d-shape-recognition
pip install -r requirements.txt
````

---

## 🧩 Estruturas de Dados

### 📁 Modelo A (via Blender)

* Imagens geradas com `blender_generator.py`.
* Organização em pastas por classe:

  ```
  data/raw/
  ├── cube/
  ├── sphere/
  ├── cone/
  └── ...
  ```
* Pré-processamento: `src/data_processing_a/`

### 📄 Modelo B (via CSV)

* Arquivo CSV define caminhos de imagens e rótulos.
* Suporta datasets externos (ex: Kaggle, personalizados).
* Pré-processamento: `src/data_processing_b/`
* Exemplo:

  ```csv
  filepath,label
  /path/to/image1.png,cube
  /path/to/image2.png,sphere
  ```

---

## 🧠 Treinamento de Modelos

### ✅ Modelo A

```python
# Código em: src/training_model_a/
```

* Otimizado para datasets gerados com Blender.
* Arquitetura única, simples e funcional.

### ✅ Modelo B — CLI Avançada e Várias Arquiteturas

```bash
# Exemplo: modelo básico
python src/training_model_b/train_model.py --architecture basic

# Modelo avançado com parâmetros customizados
python src/training_model_b/train_model.py --architecture advanced --epochs 50 --batch_size 64 --lr 0.001

# Modelo híbrido
python src/training_model_b/train_model.py --architecture hybrid
```

#### 🛠 Recursos do Modelo B:

* **Múltiplas arquiteturas disponíveis:**

  * `basic`: CNN leve para dispositivos com baixa capacidade
  * `advanced`: baseada em EfficientNet + SE Blocks
  * `hybrid`: ensemble de modelos
  * *(outros modelos podem ser adicionados com facilidade)*
* **CLI poderosa:**

  ```bash
  --architecture {basic,advanced,hybrid}
  --batch_size INT
  --lr FLOAT
  --epochs INT
  ```
* **Treinamento inteligente:**

  * Early stopping adaptativo
  * Redução automática da taxa de aprendizado
  * Checkpoints com versionamento automático de nomes

---

## 🔧 Dependências e Requisitos

### ✔️ Requisitos do sistema

* Python 3.8 ou superior
* 4GB+ de RAM (recomendado 8GB)
* GPU (opcional, mas recomendado para treinar modelos avançados)
* Blender (apenas para o Modelo A)

### 📦 Principais bibliotecas

```txt
tensorflow>=2.11.0
opencv-python
numpy
pandas
scikit-learn
matplotlib
caer
albumentations
```

Para gerar dados com Blender:

* Blender 2.9 ou superior
* Executar via CLI:

  ```bash
  blender --background --python src/data_generation/blender_generator.py
  ```

Instale todas as dependências com:

```bash
pip install -r requirements.txt
```

---

## 📜 Licença

MIT License. Veja o arquivo `LICENSE` para mais detalhes.

---

## 📎 Links Úteis

* [Documentação](docs/)
* [Relatório de Treinamento](docs/training_report.md)
* [Exemplo de Dataset](https://www.kaggle.com/...)

---

Feito por \[Seu Nome] | [GitHub](https://github.com/seu-usuario)

```

---

Se quiser, posso salvar isso diretamente como `README.md` e gerar o link de download. Deseja o arquivo pronto?
```
