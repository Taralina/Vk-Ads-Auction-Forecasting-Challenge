import streamlit as st
import pandas as pd
import lightgbm as lgb
import numpy as np
from models.data_processing import preprocessing, generate_predictions
import time
from io import StringIO

# Функция для загрузки данных
def load_data():
    history_file = st.file_uploader("Загрузите файл history.tsv", type=["tsv", "csv"])
    user_file = st.file_uploader("Загрузите файл user.tsv", type=["tsv", "csv"])
    validate_file = st.file_uploader("Загрузите файл validate.tsv", type=["tsv", "csv"])

    if history_file is not None:
        history = pd.read_csv(history_file, sep="\t")
    else:
        history = None

    if user_file is not None:
        users = pd.read_csv(user_file, sep="\t")
    else:
        users = None

    if validate_file is not None:
        validate = pd.read_csv(validate_file, sep="\t")
    else:
        validate = None

    return history, users, validate



# Главная часть Streamlit-приложения
def main():
    st.title("Прогнозирование показов рекламы")

    # Загрузка данных
    history, users, validate = load_data()

    if history is not None and users is not None and validate is not None:
        # Таймер для загрузки данных
        start_load_time = time.time()
        df = preprocessing(history, users, validate)
        st.write("Данные загружены. Нажмите кнопку, чтобы сделать предсказание.")
        load_duration = time.time() - start_load_time
        st.write(f"Время обработки файлов: {load_duration:.2f} секунд")

        
        if st.button("Сделать предсказание"):
            # Таймер для предсказания
            start_predict_time = time.time()
            # Генерация предсказаний
            predictions = generate_predictions(df)

            # Таймер для предсказания завершен
            predict_duration = time.time() - start_predict_time
            st.write(f"Время получения предсказания: {predict_duration:.2f} секунд")

            # Преобразуем предсказания в файл для скачивания
            output = StringIO()
            predictions.to_csv(output, sep='\t', index=False)
            output.seek(0)  # Возвращаем курсор в начало

            output_file = output.getvalue().encode('utf-8')

            # Скачивание результатов
            st.write("Предсказания успешно получены!")
            st.download_button('Скачать файл с предсказаниями', output_file, file_name="predictions.tsv")
# Запуск приложения
if __name__ == "__main__":
    main()