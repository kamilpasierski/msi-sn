import os
import numpy as np
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

    if request.method == 'POST' and request.FILES.getlist('image'):
        results = []
        fs = FileSystemStorage()

        for myfile in request.FILES.getlist('image'):
            try:
                # 1. Zapis pliku
                filename = fs.save(myfile.name, myfile)
                uploaded_file_url = fs.url(filename)
                file_path = os.path.join(settings.MEDIA_ROOT, filename)

                # 2. Przetwarzanie
                img = load_img(file_path, target_size=(28, 28), color_mode="grayscale")
                img_array = img_to_array(img)
                img_array = 1.0 - (img_array / 255.0)
                img_array = np.expand_dims(img_array, axis=0)

                # 3. Predykcja
                if model:
                    predictions = model.predict(img_array)
                    predicted_idx = np.argmax(predictions[0])
                    confidence = round(100 * np.max(predictions[0]), 2)
                    result_text = class_names[predicted_idx]

                    # Wynik konkretnego pliku
                    results.append({
                        'url': uploaded_file_url,
                        'prediction': result_text,
                        'confidence': confidence,
                        'filename': filename
                    })
            except Exception as e:
                print(f"Błąd przy pliku {myfile.name}: {e}")

        # Przekazujemy całą listę wyników do HTML
        context['results'] = results

    return render(request, 'core/index.html', context)