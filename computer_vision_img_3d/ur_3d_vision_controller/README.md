# 🤖 Vision UR Controller – Robô com Visão 3D

Sistema completo para controle de robô UR (Universal Robots) utilizando visão 3D com câmera Balmer/RealSense.

---

## 📊 Visão Geral

Este sistema integra:

* Captura e processamento de imagens 3D
* Detecção de objetos usando redes neurais
* Transformação de coordenadas entre câmera e robô
* Controle preciso do robô UR via interface RTDE

---

## 🔧 Componentes Principais

### 1. `DepthCamera`

* Gerencia a câmera RealSense
* Captura frames alinhados (cor + profundidade)
* Converte coordenadas de pixel para 3D

### 2. `ObjectDetector`

* Carrega modelo TensorFlow/Keras
* Detecta objetos em imagens RGB
* Retorna bounding box e posição 3D

### 3. `CoordinateTransformer`

* Converte coordenadas da câmera para o robô
* Usa matriz de calibração
* Calcula orientação do gripper com base na altura

### 4. `URController`

* Controla robô UR via protocolo RTDE
* Move para poses definidas
* Gera e executa scripts URScript

### 5. `VisionToMotionController`

* Integra todos os componentes acima
* Gerencia ciclos de detecção e movimento

---

## ⚖️ Requisitos

* Python 3.7+
* Bibliotecas:

  * OpenCV
  * pyrealsense2
  * TensorFlow 2.x
  * ur-rtde
  * NumPy

---

## ⚙️ Configuração

### 1. Configuração da câmera

```python
{
    'width': 640,
    'height': 480,
    'fps': 30
}
```

### 2. Arquivo de calibração (`calibration.json`)

```json
{
  "transformation_matrix": [...],
  "tool_offset": [0, 0, 0]
}
```

---

## ▶️ Uso Básico

```python
# Inicialização
de vision_to_motion_controller import VisionToMotionController

system = VisionToMotionController(
    camera_config=CONFIG['camera'],
    model_path='object_detector.h5',
    calibration_file='calibration.json',
    robot_ip='192.168.1.10'
)

try:
    while True:
        success = system.run_single_cycle()
        print("Sucesso" if success else "Sem detecção")
        time.sleep(1)
except KeyboardInterrupt:
    print("Parando sistema...")
finally:
    system.shutdown()
```

---

## ⚖️ Fluxo de Operação

1. Captura de frames alinhados (cor + profundidade)
2. Detecção de objeto principal
3. Conversão para coordenadas do robô
4. Cálculo das poses: aproximação, alvo, retirada
5. Geração e execução do script de movimento

---

## 🔄 Personalização

* **Modelo de detecção**: Substitua `object_detector.h5` pelo seu próprio
* **Parâmetros de movimento**: Altere velocidades no `URController`
* **Estratégia de abordagem**: Modifique `calculate_gripper_orientation`

---

## ⚠️ Limitações

* Exige calibração precisa câmera ↔ robô
* Modelo de detecção deve ser previamente treinado
* Velocidade segura por padrão (ajustável)

---

## 📈 Melhorias Futuras

* Rastreamento de objeto entre ciclos
* Verificação de colisão
* Feedback de força no pick
* Suporte a múltiplos objetos

---

## 🔗 Referências

* [Intel RealSense SDK](https://www.intelrealsense.com/sdk-2/)
* [UR RTDE Guide](https://www.universal-robots.com/articles/ur/interface-communication/real-time-data-exchange-rtde-guide/)
* [TensorFlow Keras Docs](https://www.tensorflow.org/guide/keras)
