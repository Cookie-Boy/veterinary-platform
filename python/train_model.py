import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from minio import Minio
from minio.error import S3Error

# ===== Конфигурация MinIO из переменных окружения =====
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'localhost:9000')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'minioadmin')
MINIO_BUCKET = os.getenv('MINIO_BUCKET', 'ml-models')
MINIO_MODEL_OBJECT = os.getenv('MINIO_MODEL_OBJECT', 'pet_health_model.pkl')
# =====================================================

# Загрузка датасета
DATASET_NAME = "pet_vitals_v1.csv"
MODEL_PATH = "pet_health_model.pkl"
df = pd.read_csv(DATASET_NAME)

# Признаки и целевая переменная
X = df[['species', 'breed', 'heartRate', 'respiration', 'temperature']]
y = df['anomaly_class']

# Разделение
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Предобработка
categorical_cols = ['species', 'breed']
numerical_cols = ['heartRate', 'respiration', 'temperature']

categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Модель
classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', classifier)
])

# Обучение
model.fit(X_train, y_train)

# Оценка
y_pred = model.predict(X_test)
print("Отчёт о классификации на тестовой выборке:")
print(classification_report(y_test, y_pred, target_names=['норма', 'аном.пульс', 'аном.дыхан.', 'аном.темп.']))

# Сохраняем модель временно локально
local_model_path = MODEL_PATH
joblib.dump(model, local_model_path)
print(f"Модель сохранена локально: {local_model_path}")

# Загружаем в MinIO
try:
    # Инициализируем клиент MinIO
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False  # если используете HTTP, а не HTTPS
    )

    # Проверяем/создаём бакет
    found = minio_client.bucket_exists(MINIO_BUCKET)
    if not found:
        minio_client.make_bucket(MINIO_BUCKET)
        print(f"Бакет '{MINIO_BUCKET}' создан.")
    else:
        print(f"Бакет '{MINIO_BUCKET}' уже существует.")

    # Загружаем файл
    minio_client.fput_object(
        MINIO_BUCKET,
        MINIO_MODEL_OBJECT,
        local_model_path,
        content_type='application/octet-stream'
    )
    print(f"Модель успешно загружена в MinIO: {MINIO_BUCKET}/{MINIO_MODEL_OBJECT}")

    # Опционально удаляем локальный файл
    os.remove(local_model_path)
    print("Локальный файл удалён.")

except S3Error as e:
    print(f"Ошибка MinIO: {e}")
except Exception as e:
    print(f"Общая ошибка: {e}")