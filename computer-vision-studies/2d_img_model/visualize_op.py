import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

# ========== Dados ==========
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[..., np.newaxis] / 255.0
x_test = x_test[..., np.newaxis] / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# ========== Modelo ==========
def create_model():
    input_layer = Input(shape=(28, 28, 1))
    x = Conv2D(16, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(10, activation='softmax')(x)
    return Model(inputs=input_layer, outputs=x)

# ========== Otimizadores ==========
optimizers = {
    "SGD": SGD(),
    "Adam": Adam(),
    "RMSprop": RMSprop()
}

# ========== Treinamento e Avaliação ==========
for name, optimizer in optimizers.items():
    print(f"\n🧪 Treinando com {name}...")
    model = create_model()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=3, batch_size=128, verbose=0, validation_split=0.1)
    
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"{name} -> Test Accuracy: {acc:.4f}")

# ========== Visualização dos Resultados ==========
print("\n📊 Explicação dos resultados:")
print("""
• SGD (Stochastic Gradient Descent):
  - Atualiza os pesos com passos fixos.
  - Pode demorar mais para convergir e é sensível à escala dos dados.
  - Resultado esperado: menor acurácia em poucos epochs.

• Adam (Adaptive Moment Estimation):
  - Combina RMSprop e momentum.
  - Adapta automaticamente a taxa de aprendizado para cada parâmetro.
  - Costuma convergir mais rápido e com maior precisão.

• RMSprop:
  - Adapta a taxa de aprendizado com base nas médias das derivadas quadradas.
  - Funciona bem para problemas com ruído e sinais não estacionários.
  - Também atinge alta acurácia, similar ao Adam, mas às vezes um pouco abaixo.
""")

