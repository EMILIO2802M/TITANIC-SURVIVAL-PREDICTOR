from django.conf import settings
from django.shortcuts import render
import os
import joblib
import numpy as np
from rest_framework.views import APIView
from rest_framework.response import Response

# 1. Cargar el modelo v2 (sigue siendo el mismo)
MODEL_PATH = os.path.join(settings.BASE_DIR, 'predictor/titanic_model.joblib')
model = joblib.load(MODEL_PATH)

# 2. Medias del dataset (los valores que imprimió el script anterior)
#    (Pega aquí los valores exactos que te dio train_model.py)
DEFAULT_AGE = 29.69911764705882  # (Ejemplo, usa el tuyo)

# 3. ¡VOLVEMOS A AGREGAR EL MAPA DE TARIFAS!
FARE_MAP = {
    1: 84.15,  # Tarifa media Clase 1
    2: 20.66,  # Tarifa media Clase 2
    3: 13.68   # Tarifa media Clase 3
}

class PredictView(APIView):
    """
    Vista de API que recibe: pclass, sex, age, embarked
    Y devuelve: survival_percentage
    (La tarifa se calcula internamente)
    """
    def get(self, request, *args, **kwargs):
        """
        Esta función se ejecuta cuando alguien VISITA (GET) la URL.
        Simplemente le muestra el formulario.
        """
        return render(request, 'index.html')
    
    def post(self, request, *args, **kwargs):
        try:
            # A. Obtener datos del request (SIN TARIFA)
            data = request.data
            
            pclass = int(data.get('pclass'))
            sex_str = data.get('sex')
            age_str = data.get('age')
            embarked = data.get('embarked') # 'S', 'C', o 'Q'

            # B. Procesar los datos para el modelo
            
            # Sexo
            sex_encoded = 1 if sex_str.lower() == 'female' else 0
            
            # Edad (si está vacío, usamos la media)
            age = float(age_str) if age_str else DEFAULT_AGE
            
            # ¡Tarifa (ASIGNADA AUTOMÁTICAMENTE)!
            fare = FARE_MAP.get(pclass, 13.68) # Default a 3ra clase

            # Puerto (Convertir 'S', 'C', 'Q' en 3 columnas)
            embarked_c = 1 if embarked == 'C' else 0
            embarked_q = 1 if embarked == 'Q' else 0
            embarked_s = 1 if embarked == 'S' else 0

            # C. Crear el array de características
            # ¡EL ORDEN DEBE SER IDÉNTICO AL DEL ENTRENAMIENTO!
            # ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
            features = np.array([[
                pclass, 
                sex_encoded, 
                age, 
                fare, 
                embarked_c, 
                embarked_q, 
                embarked_s
            ]])

            # D. Hacer la predicción
            probability = model.predict_proba(features)[0][1]
            
            # E. Formatear y devolver la respuesta
            survival_percentage = round(probability * 100, 2)
            
            return Response({'survival_percentage': survival_percentage})

        except Exception as e:
            return Response({'error': str(e)}, status=400)