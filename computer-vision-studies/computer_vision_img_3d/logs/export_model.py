import tensorflow as tf
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='Caminho do modelo .h5')
parser.add_argument('--out_dir', type=str, required=True, help='Diretório de saída')
args = parser.parse_args()

# Carrega modelo Keras
model = tf.keras.models.load_model(args.model_path)

# Exporta para TFLite
tflite_path = os.path.join(args.out_dir, "model.tflite")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)
print(f"✅ Modelo salvo como .tflite: {tflite_path}")

# Exporta para ONNX (requer tf2onnx)
try:
    import tf2onnx
    import subprocess

    onnx_path = os.path.join(args.out_dir, "model.onnx")
    subprocess.run([
        "python", "-m", "tf2onnx.convert",
        "--saved-model", args.model_path,
        "--output", onnx_path
    ])
    print(f"✅ Modelo salvo como .onnx: {onnx_path}")
except ImportError:
    print("⚠️ tf2onnx não instalado. Instale com: pip install tf2onnx")
