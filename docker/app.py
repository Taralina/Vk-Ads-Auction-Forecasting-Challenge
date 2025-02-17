import pandas as pd
import lightgbm as lgb
from flask import Flask, request, jsonify
from io import StringIO
from models.data_processing import preprocessing, generate_predictions

# Инициализируем Flask приложение
app = Flask(__name__)


# Загружаем модели
model_1 = lgb.Booster(model_file='models/model_1.txt')
model_2 = lgb.Booster(model_file='models/model_2.txt')
model_3 = lgb.Booster(model_file='models/model_3.txt')

def process_files(history_file, users_file, validate_file):
    # Загружаем данные из CSV файлов
    history = pd.read_csv(StringIO(history_file))
    users = pd.read_csv(StringIO(users_file))
    validate = pd.read_csv(StringIO(validate_file))

    # Применяем препроцессинг
    train_df = preprocessing(history, users, validate)

    # Генерация предсказаний
    predictions = generate_predictions(train_df)

    return predictions

@app.route('/predict', methods=['POST'])
def predict():
    # Получаем файлы из POST-запроса
    history_file = request.files['history'].read().decode('utf-8')
    users_file = request.files['users'].read().decode('utf-8')
    validate_file = request.files['validate'].read().decode('utf-8')

    # Обрабатываем файлы и получаем предсказания
    predictions = process_files(history_file, users_file, validate_file)

    # Конвертируем результат в JSON
    predictions_json = predictions.to_dict(orient='records')

    # Возвращаем предсказания как JSON
    return jsonify(predictions_json)

if __name__ == "__main__":
    # Запускаем Flask API
    app.run(debug=True, host='0.0.0.0', port=5000)
