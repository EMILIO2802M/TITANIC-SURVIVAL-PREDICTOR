import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

print("Iniciando entrenamiento del modelo v2 (con Tarifa y Puerto)...")

# Cargar el dataset
try:
    data = pd.read_csv('train.csv')
except FileNotFoundError:
    print("Error: No se encontró el archivo train.csv en la carpeta /notebook/")
    exit()

# --- 1. Preprocesamiento de Datos ---

# Usaremos una copia para evitar warnings
df = data.copy()

# Rellenar valores faltantes (NaN)
# Usamos 'S' para el puerto (es el más común, la 'moda')
df['Embarked'] = df['Embarked'].fillna('S') 

# Rellenar 'Age' y 'Fare' con su promedio (media)
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

# Convertir 'Sex' a números (0 o 1)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Convertir 'Embarked' a números (One-Hot Encoding)
# Esto crea 3 nuevas columnas: Embarked_S, Embarked_C, Embarked_Q
df = pd.get_dummies(df, columns=['Embarked'], drop_first=False)

# --- 2. Definición de Características ---

# Esta es nuestra NUEVA lista de features
features = [
    'Pclass', 
    'Sex', 
    'Age', 
    'Fare', 
    'Embarked_C', # 1 si es Cherbourg, 0 si no
    'Embarked_Q', # 1 si es Queenstown, 0 si no
    'Embarked_S'  # 1 si es Southampton, 0 si no
]
target = 'Survived'

# Asegurarnos de que todas las columnas existan
# (a veces si un valor no aparece, la columna no se crea, ej: Embarked_Q)
for col in ['Embarked_C', 'Embarked_Q', 'Embarked_S']:
    if col not in df.columns:
        df[col] = 0

X = df[features]
y = df[target]

# --- 3. Entrenamiento ---

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X, y)

# --- 4. Guardado ---
output_dir = '../backend/predictor'
model_path = os.path.join(output_dir, 'titanic_model.joblib')
joblib.dump(model, model_path)

print(f"¡Modelo v2 entrenado y guardado exitosamente en: {model_path}!")
print("--- Medias para usar en la API (copiar en views.py) ---")
print(f"DEFAULT_AGE = {df['Age'].mean()}")
print(f"DEFAULT_FARE = {df['Fare'].mean()}")