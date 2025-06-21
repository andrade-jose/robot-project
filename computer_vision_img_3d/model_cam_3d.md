Aqui está o arquivo README.md para o código fornecido:

```markdown
# Vision UR Controller

Sistema completo para controle de robô UR (Universal Robots) utilizando visão 3D com câmera Balmer/RealSense.

## Visão Geral

Este sistema integra:
- Captura e processamento de imagens 3D
- Detecção de objetos usando redes neurais
- Transformação de coordenadas entre câmera e robô
- Controle preciso do robô UR via interface RTDE

## Componentes Principais

### 1. DepthCamera
Classe responsável por:
- Configurar e gerenciar a câmera RealSense
- Capturar frames alinhados de cor e profundidade
- Converter coordenadas de pixel para coordenadas 3D

### 2. ObjectDetector
Classe que implementa:
- Carregamento de modelo de detecção pré-treinado (TensorFlow/Keras)
- Detecção de objetos em imagens coloridas
- Retorno de posições 3D e bounding boxes

### 3. CoordinateTransformer
Classe para:
- Carregar matriz de transformação de calibração
- Converter coordenadas entre sistemas de referência (câmera → robô)
- Calcular orientação adequada para o gripper

### 4. URController
Classe de controle do robô UR que oferece:
- Conexão via RTDE (Real-Time Data Exchange)
- Movimento para poses específicas
- Geração e execução de scripts URScript
- Sequências de pick-and-place

### 5. VisionToMotionController
Classe principal que integra todos os componentes:
- Atualização do sistema de visão
- Cálculo de poses para operação
- Execução de sequências completas
- Gerenciamento do ciclo de detecção-movimento

## Requisitos

- Python 3.7+
- Bibliotecas:
  - OpenCV
  - PyRealSense2
- TensorFlow 2.x
- ur-rtde
- NumPy

## Configuração

1. Arquivo de configuração da câmera:
```python
{
    'width': 640,   # Largura da imagem
    'height': 480,  # Altura da imagem
    'fps': 30       # Taxa de quadros
}
```

2. Arquivo de calibração (`calibration.json`):
```json
{
    "transformation_matrix": [...],  # Matriz 4x4 de transformação
    "tool_offset": [0, 0, 0]        # Offset da ferramenta
}
```

## Uso Básico

```python
# Inicialização
system = VisionToMotionController(
    camera_config=CONFIG['camera'],
    model_path='object_detector.h5',
    calibration_file='calibration.json',
    robot_ip='192.168.1.10'
)

try:
    # Loop principal
    while True:
        success = system.run_single_cycle()
        time.sleep(1)
        
except KeyboardInterrupt:
    print("Parando sistema...")
finally:
    system.shutdown()
```

## Fluxo de Operação

1. Captura de frames alinhados (cor + profundidade)
2. Detecção de objetos e cálculo da posição 3D
3. Transformação para coordenadas do robô
4. Cálculo das poses (aproximação, target, retirada)
5. Geração e execução do script de movimento

## Personalização

- **Modelo de detecção**: Substitua `object_detector.h5` por seu modelo customizado
- **Parâmetros de movimento**: Ajuste velocidades e acelerações nos métodos de movimento
- **Orientação do gripper**: Modifique `calculate_gripper_orientation` para diferentes estratégias

## Limitações

- Requer calibração precisa entre câmera e robô
- Modelo de detecção precisa ser treinado para objetos específicos
- Velocidades de movimento conservadoras por padrão (segurança)

## Melhorias Futuras

- Implementar tracking de objetos entre frames
- Adicionar verificação de colisões
- Integrar feedback de força durante o pick
- Suporte a múltiplos objetos e planejamento de trajetória
```