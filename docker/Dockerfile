# Используйте базовый образ Python
FROM python:3.9

# Установка зависимостей
RUN apt-get update && \
    apt-get install -y libgomp1

# Установка Python библиотек
COPY requirements.txt /app/
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

# Копирование файлов с кодом
COPY . /app

# Открытие порта
EXPOSE 5000

# Команда для запуска приложения
CMD ["python", "app.py"]
