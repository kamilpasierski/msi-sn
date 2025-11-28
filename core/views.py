import os
import numpy as np
import tensorflow as tf
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --- ŁADOWANIE MODELU ---
MODEL_PATH = os.path.join(settings.BASE_DIR, 'core', 'my_model.keras')

try:
    print(f"Próba załadowania modelu z: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print("Model załadowany pomyślnie!")
except Exception as e:
    print(f"BŁĄD: Nie udało się załadować modelu. Sprawdź ścieżkę! {e}")
    model = None

# Nazwy klas (kolejność musi być taka sama jak w Fashion MNIST)
class_names = ['T-shirt/Top', 'Spodnie', 'Sweter', 'Sukienka', 'Płaszcz',
               'Sandał', 'Koszula', 'Tenisówka', 'Torba', 'But']


def index(request):
    context = {}

    # Jeśli użytkownik wysłał formularz (POST) i jest w nim plik (image)
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            # 1. Zapisanie pliku tymczasowo na dysku
            myfile = request.FILES['image']
            fs = FileSystemStorage()
            filename = fs.save(myfile.name, myfile)
            uploaded_file_url = fs.url(filename)
            file_path = os.path.join(settings.MEDIA_ROOT, filename)

            # 2. Przygotowanie zdjęcia dla sieci neuronowej
            # Sieć uczyła się na obrazkach 28x28, czarno-białych (grayscale)
            img = load_img(file_path, target_size=(28, 28), color_mode="grayscale")

            # Zamiana obrazka na tablicę liczb
            img_array = img_to_array(img)

            # Normalizacja i inwersja kolorów
            # (MNIST ma białe ubrania na czarnym tle, zdjęcia zazwyczaj odwrotnie)
            img_array = 1.0 - (img_array / 255.0)

            # Dodanie wymiaru (Keras oczekuje "paczki" zdjęć, nawet jak jest jedno)
            img_array = np.expand_dims(img_array, axis=0)

            # 3. Predykcja (Wnioskowanie)
            if model:
                predictions = model.predict(img_array)
                predicted_idx = np.argmax(predictions[0])  # Indeks najwyższego wyniku
                confidence = round(100 * np.max(predictions[0]), 2)  # Pewność w %

                result_text = class_names[predicted_idx]

                # Przekazujemy wyniki do HTMLa
                context = {
                    'uploaded_file_url': uploaded_file_url,
                    'result': result_text,
                    'confidence': confidence
                }
            else:
                context['error'] = "Błąd: Model nie jest załadowany."

        except Exception as e:
            context['error'] = f"Wystąpił błąd podczas przetwarzania: {e}"

    return render(request, 'core/index.html', context)