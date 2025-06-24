Parâmetros de Treinamento por Linha de Comando (Modelo B)

Exemplo de execução:

python src/training_model_b/train_model.py \
  --architecture advanced \
  --epochs 50 \
  --batch_size 64 \
  --lr 0.001

Parâmetros Disponíveis

--architecture: Define a arquitetura a ser usada: basic, advanced, hybrid

--epochs: Quantidade de épocas de treinamento (padrão: 30)

--batch_size: Tamanho do lote (padrão: 32)

--lr: Taxa de aprendizado (padrão: 0.0005)

--dataset: Caminho para CSV do dataset (opcional)

--save_dir: Caminho onde o modelo treinado será salvo

Esse sistema de configuração permite automatizar experiências e comparar resultados de forma rápida.