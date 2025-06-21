# 📷🤖 Sistema Vision-to-Motion com Visão Estéreo + RTDE (UR)

Este projeto implementa um sistema completo de visão computacional com **duas câmeras RGB simples** (visão estéreo) integrado com o controle de um robô **UR** via **RTDE**.

---

## 🔧 Componentes Principais

### `StereoCamera`

Classe responsável por capturar imagens das duas câmeras e calcular profundidade.

**Funções:**

- `get_frames()` — Captura imagens esquerda e direita.
- `compute_depth_map(left, right)` — Gera mapa de disparidade usando `StereoBM`.
- `disparity_to_depth(disparity)` — Converte disparidade para profundidade.
- `pixel_to_3d(u, v, disparity)` — Converte um pixel em coordenadas 3D.

> 📌 Parâmetros de calibração são fixos, mas podem ser carregados de um arquivo.

---

### `ObjectDetector`

Utiliza um modelo Keras para detectar um objeto na imagem da câmera esquerda.

**Funções:**

- `__init__(model_path)` — Carrega o modelo `.h5`.
- `detect(image)` — Retorna `(u, v)` do centro do objeto, se detectado.

> ⚠️ Assume saída com bounding box `[x1, y1, x2, y2, conf]`.

---

### `CoordinateTransformer`

Transforma coordenadas da câmera para o sistema do robô.

**Funções:**

- `__init__(calibration_file)` — Carrega matriz de transformação 4x4 e offset.
- `camera_to_robot(pos)` — Aplica a matriz para obter posição no robô.
- `get_orientation(height)` — Define orientação baseada na altura do objeto.

---

### `URController`

Controlador do robô UR com RTDE.

**Funções:**

- `__init__(robot_ip, config_file)` — Carrega `rtde_config.xml` e conecta.
- `send_pose(pose)` — Envia pose `[x, y, z, rx, ry, rz]`.
- `disconnect()` — Finaliza a conexão com o robô.

> 📎 Precisa dos arquivos `rtde_config.xml` configurados com recipes (`setp`, `state`).

---

### `VisionToMotion`

Integra todos os módulos. Executa os ciclos de captura, detecção, triangulação e controle.

**Funções:**

- `run_cycle()` — Executa:
  1. Captura de imagens
  2. Detecção do objeto
  3. Conversão para coordenadas do robô
  4. Envio de comandos de movimento
- `shutdown()` — Libera recursos.

---

## 📁 Requisitos

- `object_detector.h5` — Modelo treinado
- `calibration.json` — Matriz de transformação 4x4
- `rtde_config.xml` — Recipes do robô UR

---

## ▶️ Execução

```bash
python vision_ur_controller.py
