# **Sistema de Reconhecimento de Formas 3D**  

Um projeto completo para classificação de formas geométricas 3D usando **visão computacional** e **deep learning**, incluindo geração de dados sintéticos, treinamento de modelo CNN e inferência em tempo real.  

---

## **📦 Estrutura do Projeto**  
```
3d_shape_recognition/  
├── data/  
│   ├── samples/          # 📂 Exemplos de imagens (10 por classe)  
│   ├── raw/              # ⚠️ Pasta para dados brutos (não versionada)  
│   └── processed/        # ⚠️ Dados pré-processados (não versionada)  
├── models/  
│   ├── trained/          # 🏋️ Modelos treinados (ex: best_model.h5)  
│   └── architectures/    # 📝 Definições de arquitetura  
├── src/  
│   ├── data_generation/  # 🌀 Scripts do Blender para gerar dados  
│   ├── data_processing/  # 🔄 Pré-processamento de imagens  
│   ├── training/         # 🧠 Treinamento do modelo  
│   ├── inference/        # 🔮 Predição em tempo real  
│   └── utils/            # 🛠️ Funções auxiliares  
├── tests/                # 🧪 Testes unitários  
├── docs/                 # 📚 Documentação adicional  
├── requirements.txt      # 📜 Dependências do Python  
└── README.md             # 🏁 Este arquivo  
```  

---

## **🚀 Como Usar**  

### **1️⃣ Instalação**  
```bash
git clone https://github.com/seu-usuario/3d-shape-recognition.git  
cd 3d-shape-recognition  
pip install -r requirements.txt  
```  

### **2️⃣ Obter o Dataset**  
O dataset completo **não está incluído** no repositório devido ao tamanho.  

#### **Opção A: Baixar dataset pré-gerado**  
📥 [Google Drive](https://drive.google.com/...) | [Kaggle](https://www.kaggle.com/...)  
```bash
# Extraia em data/raw/
unzip dataset_shapes_3d.zip -d data/raw/
```  

#### **Opção B: Gerar dados com Blender (recomendado para customização)**  
```bash
blender --background --python src/data_generation/blender_generator.py
```  

### **3️⃣ Treinar o Modelo**  
```python
from src.training.trainer import ShapeRecognizerTrainer  

trainer = ShapeRecognizerTrainer()  
model, history = trainer.train(train_gen, val_gen, epochs=50)  
```  

### **4️⃣ Executar Reconhecimento em Tempo Real**  
```python
from src.inference.realtime_predictor import RealTimePredictor  

predictor = RealTimePredictor("models/trained/best_model.h5", ["cubo", "esfera", "cone"])  
predictor.run()  # 🎥 Abre a webcam e classifica formas!
```  

---

## **📌 Dados e Versionamento**  
- **Dataset pequeno**: `data/samples/` contém exemplos para teste rápido.  
- **Dataset completo**: Baixe separadamente ou gere via Blender.  
- **Git LFS**: Se precisar versionar dados grandes, use:  
  ```bash
  git lfs install
  git lfs track "data/raw/*.png"
  git add .gitattributes
  ```  

---

## **🛠 Dependências**  
- Python 3.8+  
- TensorFlow 2.x  
- OpenCV  
- NumPy  
- Caer  
- Blender (para geração de dados)  

Instale tudo via:  
```bash
pip install -r requirements.txt
```  

---

## **📜 Licença**  
MIT License. Consulte `LICENSE` para detalhes.  

---

## **📎 Links Úteis**  
- [Documentação Completa](docs/)  
- [Relatório de Treinamento](docs/training_report.md)  
- [Exemplo de Dataset no Kaggle](https://www.kaggle.com/...)  

--- 

Feito por [Seu Nome] | [GitHub](https://github.com/seu-usuario) | [LinkedIn](...)  

🔹 **Dúvidas?** Abra uma *issue* ou entre em contato!