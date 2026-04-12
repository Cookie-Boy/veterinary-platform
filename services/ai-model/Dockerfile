FROM python:3.11-slim

WORKDIR /app

# Устанавливаем системные зависимости (если нужны для pandas/numpy)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Копируем и устанавливаем Python зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь код сервиса
COPY . .

# Создаём директорию для временных файлов (если нужно)
RUN mkdir -p /tmp

# Переменные окружения (можно переопределить в docker-compose)
ENV MINIO_ENDPOINT=minio:9000
ENV MINIO_ACCESS_KEY=minioadmin
ENV MINIO_SECRET_KEY=minioadmin
ENV MINIO_BUCKET=ml-models
ENV MINIO_MODEL_OBJECT=pet_health_model.pkl

# Порт для Flask (по умолчанию 5000, но можно изменить)
EXPOSE 5000

# Запуск приложения
CMD ["python", "app.py"]