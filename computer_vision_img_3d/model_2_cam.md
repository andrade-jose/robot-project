# vision_ur_controller_standard.py

Sistema de visão computacional para controle de robôs Universal Robots (UR) utilizando câmeras convencionais com estereoscopia para estimativa de profundidade.

---

## Objetivo

Permitir que um robô UR identifique objetos em seu ambiente utilizando um par de câmeras estéreo, estimando profundidade para realizar ações coordenadas e precisas.

---

## Estrutura do Código

### StereoCamera

- Inicializa e gerencia dois dispositivos de captura de vídeo (câmeras esquerda e direita).
- Configura resolução e parâmetros básicos.
- Carrega parâmetros de calibração estéreo a partir de arquivo JSON (`stereo_calibration.json`).
- Calcula disparidade entre imagens e gera mapa de profundidade.
- Métodos principais:
  - `get_frames()`: captura imagens sincronizadas das duas câmeras e retorna imagens e mapa de profundidade.
  - `stop()`: libera os dispositivos de vídeo.

### ObjectDetector

- Detecta objetos a partir da imagem colorida e do mapa de profundidade.
- Adaptado para usar mapas de profundidade estimados por estereoscopia.
- Retorna as informações do objeto detectado (posição, tamanho, etc).

### CoordinateTransformer

- Transforma coordenadas da câmera para o referencial do robô UR.
- Utiliza filtro de Kalman para suavizar as medidas e compensar ruídos.
- Aplica transformação homogênea (matriz 4x4) para converter coordenadas.

### VisionToMotionController

- Classe principal para integrar captura de imagens, detecção de objetos e controle do robô.
- Não implementada no trecho fornecido, mas deve orquestrar os processos de captura → detecção → movimentação.

### calibrate_stereo

- Função para calibração offline do sistema estéreo.
- Deve ser implementada para detectar cantos em imagens, calcular parâmetros intrínsecos e extrínsecos e gerar a matriz Q.

---

## Configuração Exemplo

```python
STEREO_CONFIG = {
    'cameras': {
        'left_index': 0,
        'right_index': 1,
        'width': 640,
        'height': 480
    },
    'calibration_file': 'stereo_calibration.json',
}
