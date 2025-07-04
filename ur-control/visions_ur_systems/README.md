# Vision UR Systems

Sistema modular para controle de robôs UR (Universal Robots) utilizando diferentes abordagens de visão computacional:

* 🌍 **Câmera Única com estimativa de profundidade** (MonoDepth/MiDaS)
* 👀 **Visão Estéreo com triangulação simples**
* 🔬 **Visão Estéreo com calibração precisa e matriz Q**

Cada abordagem é encapsulada em seu próprio módulo com arquivos de execução independentes.

---

## Estrutura do Projeto

```bash
vision_ur_systems/
├── main_mono.py                # Execução com câmera única
├── main_stereo_simple.py       # Execução com estéreo básico
├── main_stereo_calibrated.py   # Execução com estéreo calibrado
│
├── mono/
│   ├── mono_camera.py
│   └── vision_to_motion.py
│
├── stereo_simple/
│   ├── stereo_camera.py
│   └── vision_to_motion.py
│
├── stereo_calibrated/
│   ├── stereo_camera.py
│   ├── calibration_utils.py
│   └── vision_to_motion.py
│
├── detection/
│   └── object_detector.py
│
├── control/
│   ├── coordinate_transformer.py
│   └── ur_controller.py
│
├── config/
│   ├── mono_calibration.json
│   ├── stereo_calibration.json
│   └── rtde_config.xml
│
├── models/
│   ├── object_detector.h5
│   └── monodepth.h5
│
└── requirements.txt
```

---

## Modos de Execução

### 1. MonoCamera (Profundidade por IA)

```bash
python main_mono.py
```

* Requer: `monodepth.h5`, `object_detector.h5`, `mono_calibration.json`

### 2. StereoCamera Simples (sem calibração)

```bash
python main_stereo_simple.py
```

* Requer: Par de câmeras, `object_detector.h5`, `calibration.json`

### 3. StereoCamera Calibrado (com matriz Q)

```bash
python main_stereo_calibrated.py
```

* Requer: Par de câmeras, `object_detector.h5`, `stereo_calibration.json`

---

## Dependências

Instale as dependências com:

```bash
pip install -r requirements.txt
```

**Requisitos principais:**

* OpenCV
* TensorFlow
* NumPy
* ur-rtde
* pyrealsense2 (opcional para sensores RealSense)

---

## Fluxo Operacional (padrão)

1. Captura de imagem (colorida e profundidade)
2. Detecção de objeto com rede neural (modelo `.h5`)
3. Cálculo da posição 3D (via triangulação, rede ou reprojectImageTo3D)
4. Conversão para sistema do robô (via matriz de transformação)
5. Cálculo de orientação e envio de comandos RTDE

---

## Customização

* **Modelos:** substitua `object_detector.h5` ou `monodepth.h5`
* **Parâmetros de movimento:** altere acelerações/velocidades em `URController`
* **Câmeras:** configure `mono_camera.py` ou `stereo_camera.py`

---

## Sugestões Futuras

* Planejamento de trajetória com verificação de colisão
* Detecção de múltiplos objetos
* Interface gráfica (UI) para configuração
* Integração com controle por feedback (força ou torque)

---

## Autor

Desenvolvido por \José Alexadre de Andrade