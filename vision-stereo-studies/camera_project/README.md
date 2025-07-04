```markdown
# Stereo Vision Robot System

Este projeto implementa um pipeline completo de visão estéreo para reconstrução de profundidade, voltado para aplicações em robótica. Utiliza câmeras estéreo calibradas para gerar mapas de disparidade, profundidade e nuvens de pontos 3D.

---

## 📁 Estrutura do Projeto

```

camera_project/
├── calibration_data/           # Onde será salvo o arquivo calibration.pkl
├── filtered/                   # Pares de imagens retificadas para calibração ou reconstrução
│   ├── left/
│   └── right/
├── output/                     # Imagens de disparidade e profundidade processadas
├── stereo_vision/             # Módulos Python organizados por responsabilidade
│   ├── calibration.py
│   ├── capture.py
│   ├── disparity.py
│   ├── reconstruction.py
│   ├── processor.py
│   └── utils.py
└── scripts/                    # Scripts de uso direto
├── calibrate.py
├── process_all_pairs.py
└── live_demo.py

````

---

## ⚙️ Requisitos

- Python 3.8+
- OpenCV (>=4.5)
- NumPy

Instale com:

```bash
pip install opencv-python numpy
````

---

## 🎯 Fluxo de Uso

### 1. 🔧 Calibrar o sistema estéreo

Prepare 10 a 25 pares de imagens com um tabuleiro de xadrez visível em ambas as câmeras.

Execute:

```bash
python scripts/calibrate.py
```

Isso gera `calibration.pkl` com os parâmetros internos e mapas de retificação.

---

### 2. 🧠 Processar imagens em lote

Coloque seus pares de imagens em `filtered/left/` e `filtered/right/`.

Depois, execute:

```bash
python scripts/process_all_pairs.py
```

Saída: mapas de disparidade e profundidade em `output/`.

---

### 3. 📹 Executar ao vivo com câmeras

Com as câmeras conectadas, rode:

```bash
python scripts/live_demo.py
```

Use `ESC` para encerrar.

---

## 🧠 Futuro: Deep Learning

O projeto foi estruturado em **classes modulares** para futura integração com redes neurais:

* Substituir `StereoSGBM` por um modelo como **PSMNet** ou **RAFT-Stereo**
* Treinar em datasets sintéticos ou capturados
* Usar `depth_map` ou `point_cloud` como entrada para navegação autônoma

---

## 🤖 Aplicações Robóticas

* Reconstrução de ambiente para navegação
* Detecção de obstáculos em 3D
* Estimativa de profundidade em ambientes reais com visão estéreo

---

## 📌 Créditos

Desenvolvido por José Alexandre De Andrade com suporte da arquitetura modular OpenCV + Python.

```