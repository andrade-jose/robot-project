# Visão UR Control - Versão Personalizada

## Visão Geral do Meu Setup

Este sistema foi adaptado para funcionar com:
- Minha câmera Logitech C920
- Meu robô UR3e (192.168.1.50)
- Meu PC com GPU NVIDIA GTX 1660 Ti
- Ambiente de trabalho com iluminação controlada

## Configuração Pessoal

```python
MY_CONFIG = {
    'camera': {
        'index': 2,   # USB port 3 no meu PC
        'width': 1280, # Resolução máxima da minha câmera
        'height': 720
    },
    'model_path': 'my_custom_model.h5',  # Modelo treinado nos meus objetos
    'calibration_file': 'my_workspace_calib.json',
    'robot_ip': '192.168.1.50',
    'workspace_limits': {  # Área de trabalho segura
        'x': [-0.5, 0.5],
        'y': [-0.3, 0.3],
        'z': [0.1, 0.6]
    }
}
```

## Meus Objetivos Principais

1. **Pick-and-place** de pequenas peças eletrônicas
2. Precisão requerida: ±2mm
3. Velocidade: 10-15 ciclos por minuto

## Adaptações que Implementei

### Melhorias no Código
- Adicionei verificação de limites de workspace
- Implementei um filtro de média móvel para coordenadas
- Ajustei os parâmetros de movimento para peças pequenas

```python
# Meu filtro personalizado
class MySmoothingFilter:
    def __init__(self, window_size=5):
        self.buffer = []
        self.window = window_size
        
    def smooth(self, coords):
        self.buffer.append(coords)
        if len(self.buffer) > self.window:
            self.buffer.pop(0)
        return np.mean(self.buffer, axis=0)
```

### Arquivos Customizados
- `my_custom_model.h5`: Modelo treinado especificamente para:
  - Placas de circuito
  - Componentes SMD
  - Conectores
- `my_workpace_calib.json`: Calibração exata para minha bancada

## Meu Fluxo de Trabalho

1. Inicializar sistema com `my_config.json`
2. Posicionar peça na área de visão
3. Executar ciclo automático
4. Verificar logs de precisão
5. Ajustar parâmetros se necessário

## Dicas Pessoais

1. **Iluminação**: Uso duas luzes LED de 6000K em ângulos de 45°
2. **Calibração**: Recalibro a cada 2 semanas ou quando mudo a câmera de posição
3. **Manutenção**: Limpo a lente da câmera diariamente

## Troubleshooting

Problemas que já enfrentei e como resolvi:

| Problema               | Solução                          |
|------------------------|----------------------------------|
| Profundidade imprecisa | Ajustei o threshold do modelo    |
| Movimentos bruscos     | Reduzi velocidade para 0.15      |
| Falsos positivos       | Aumentei confiança mínima para 0.7|

## Próximas Melhorias

1. Integrar com meu sistema de visão noturna para trabalhos em baixa luz
2. Adicionar verificação por força para peças muito pequenas
3. Criar interface gráfica personalizada

## Contatos Úteis

- Suporte UR: ur-support@mycompany.com
- Especialista em visão: vision-expert@mycompany.com
- Meu contato: my.name@mycompany.com

```

Este README personalizado inclui:
- Configurações específicas do meu setup
- Adaptações que fiz no código original
- Fluxo de trabalho personalizado
- Soluções para problemas que encontrei
- Planos de melhoria futura

Mantive uma estrutura prática com foco nas informações mais relevantes para meu uso diário, incluindo tabelas e snippets de código que consulto frequentemente.