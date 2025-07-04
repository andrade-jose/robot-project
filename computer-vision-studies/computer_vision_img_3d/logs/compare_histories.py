import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_history(file_path):
    if file_path.endswith(".npy"):
        return np.load(file_path, allow_pickle=True).item()
    elif file_path.endswith(".csv"):
        return pd.read_csv(file_path).to_dict(orient='list')
    else:
        raise ValueError(f"Formato não suportado: {file_path}")


def plot_histories(histories, labels):
    metrics = ['accuracy', 'val_accuracy', 'loss', 'val_loss']
    plt.figure(figsize=(16, 10))

    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        for hist, label in zip(histories, labels):
            if metric in hist:
                plt.plot(hist[metric], label=f"{label}")
        plt.title(metric.replace('_', ' ').title())
        plt.xlabel('Epoch')
        plt.ylabel(metric.title())
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.suptitle("Comparação de históricos de treinamento", fontsize=16, y=1.02)
    plt.show()


def print_best_metrics(histories, labels):
    print("\n📈 Melhores métricas por modelo:\n")
    for hist, label in zip(histories, labels):
        acc = max(hist.get('val_accuracy', [0]))
        top3 = max(hist.get('val_top3_accuracy', [0]))
        top5 = max(hist.get('val_top5_accuracy', [0]))
        print(f"[{label}]")
        print(f"  🔹 Melhor val_accuracy: {acc:.4f}")
        print(f"  🔹 Melhor top-3 accuracy: {top3:.4f}")
        print(f"  🔹 Melhor top-5 accuracy: {top5:.4f}")
        print("")


def main():
    parser = argparse.ArgumentParser(description="Comparar históricos de treino (.npy ou .csv)")
    parser.add_argument("paths", nargs="+", help="Caminhos para os arquivos de histórico (.npy ou .csv)")
    args = parser.parse_args()

    histories = []
    labels = []

    for path in args.paths:
        label = os.path.splitext(os.path.basename(path))[0]
        try:
            history = load_history(path)
            histories.append(history)
            labels.append(label)
        except Exception as e:
            print(f"Erro ao carregar {path}: {e}")

    if histories:
        plot_histories(histories, labels)
        print_best_metrics(histories, labels)
    else:
        print("Nenhum histórico válido foi carregado.")


if __name__ == "__main__":
    main()
