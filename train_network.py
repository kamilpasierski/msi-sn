import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

if not os.path.exists('core'):
    os.makedirs('core')

print("--- Pobieranie danych ---")
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Normalizacja (0-255 -> 0-1)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Dodanie wymiaru kanału (dla Keras wymaga to formatu: (ilość, wys, szer, kanały))
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Data Augmentation (Utrudnianie nauki dla lepszych efektów)
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),  # Odbicie lustrzane
    layers.RandomRotation(0.1),       # Obrót o max 10% (ok. 36 stopni)
    layers.RandomZoom(0.1),           # Przybliżenie/oddalenie o 10%
])

# 2. Budowa modelu (CNN)
model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),

    data_augmentation,

    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax"),
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Konfiguracja Early Stopping
early_stopping = EarlyStopping(
    monitor='val_loss',         # Obserwuj błąd na zbiorze walidacyjnym
    patience=5,                 # Czekaj 5 epok na poprawę
    restore_best_weights=True,  # Na koniec wgraj NAJLEPSZY model, a nie ostatni
    verbose=1
)

EPOCHS = 20

# 3. Trenowanie
print("--- Rozpoczynanie treningu (" + EPOCHS + "epok) ---")
model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=EPOCHS,
    validation_split=0.1,
    callbacks=[early_stopping]
)

# 4. Zapisywanie modelu w folderze aplikacji 'core'
save_path = os.path.join('core', 'my_model.keras')
model.save(save_path)

print(f"--- SUKCES! Model zapisany w: {save_path} ---")