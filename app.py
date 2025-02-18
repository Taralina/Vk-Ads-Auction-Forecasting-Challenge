import streamlit as st
import pandas as pd
import lightgbm as lgb
import numpy as np
from models.data_processing import preprocessing, generate_predictions
from models.metrics import get_smoothed_log_accuracy_ratio
import time
from io import StringIO


def generate_count(campaign_grouped_predictions, train_df):
    # Вычисляем предсказанное количество просмотров по каждой метрике
    count_at_least_one = train_df['audience_size'] * campaign_grouped_predictions['at_least_one']
    count_at_least_two = train_df['audience_size'] * campaign_grouped_predictions['at_least_two']
    count_at_least_three = train_df['audience_size'] * campaign_grouped_predictions['at_least_three']

    # Создаем новый DataFrame с результатами
    result_df = campaign_grouped_predictions.copy()
    result_df['count_at_least_one'] = count_at_least_one
    result_df['count_at_least_two'] = count_at_least_two
    result_df['count_at_least_three'] = count_at_least_three
    
    # Добавляем столбец с суммой трех метрик
    result_df['total_count_predictions'] = count_at_least_one + count_at_least_two + count_at_least_three

    return result_df


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
    st.title("Vk Ads Auction Forecasting Challenge")
    st.write(
        """ 
        Этот инструмент помогает вам предсказать количество показов рекламы на основе загруженных данных. 
        Просто загрузите три файла:

        - **История** (`history`): данные о прошлых показах рекламы.
        - **Пользователи** (`users`): информация о пользователях.
        - **Рекламные данные** (`validate`): реклама для которой надо сделать предсказание

        После того как предсказания будут выполнены, вы сможете скачать результаты в виде файла. Если у вас есть файл с реальными ответами, загрузите его, и сервис автоматически вычислит метрику точности предсказаний.

        Используйте этот сервис для оптимизации рекламных кампаний и улучшения точности ваших прогнозов.
        """
    )

    # Загрузка данных
    history, users, validate = load_data()

    if history is not None and users is not None and validate is not None:
        # Таймер для загрузки данных
        start_load_time = time.time()
        df = preprocessing(history, users, validate)
        st.write("Данные загружены. Нажмите кнопку, чтобы сделать предсказание.")
        load_duration = time.time() - start_load_time
        st.write(f"Время обработки файлов: {load_duration:.2f} секунд")

        # Загружаем файл с ответами один раз
        st.write('Если у вас есть готовые данные загрузите их для получения метрики Smoothed Mean Log Accuracy Ratio')
        answers_file = st.file_uploader("Загрузите файл с правильными ответами", type=["tsv", "csv"])
        answers = None
        if answers_file is not None:
            answers = pd.read_csv(answers_file, sep="\t")

        # Кешируем функцию для предсказаний
        @st.cache_data
        def generate_and_cache_predictions(df):
            return generate_predictions(df)

        if st.button("Сделать предсказание"):
            # Таймер для предсказания
            start_predict_time = time.time()
            # Генерация предсказаний
            predictions = generate_and_cache_predictions(df)

            # Таймер для предсказания завершен
            predict_duration = time.time() - start_predict_time
            st.write(f"Время получения предсказания: {predict_duration:.2f} секунд")

            # Расчет метрики, если файл с ответами был загружен
            if answers is not None:
                # Проверим, есть ли необходимые столбцы в данных
                required_columns = ['at_least_one', 'at_least_two', 'at_least_three']
                if all(col in answers.columns for col in required_columns) and all(col in predictions.columns for col in required_columns):
                    metric = get_smoothed_log_accuracy_ratio(predictions, answers)
                    df = pd.DataFrame(metric.items(), columns=["Метрика", "Значение (%)"])
                    st.table(df)
                    mean_metric = sum(metric.values()) / len(metric)
                    st.write(f"Общая метрика для всех предсказаний: {mean_metric:.2f}%")
    

                else:
                    st.warning("В данных отсутствуют необходимые столбцы для расчета метрики.")

            # Преобразуем предсказания в файл для скачивания
            output = StringIO()
            predictions.to_csv(output, sep='\t', index=False)
            output.seek(0)  # Возвращаем курсор в начало

            output_file = output.getvalue().encode('utf-8')

            #считаем часть аудитории в людях
            data = generate_count(predictions, validate)
            data['audience_size'] = validate['audience_size']
            data['count_at_least_one'] = np.floor(data['count_at_least_one']).astype(int)
            data['count_at_least_two'] = np.floor(data['count_at_least_two']).astype(int)
            data['count_at_least_three'] = np.floor(data['count_at_least_three']).astype(int)
            data['total_count_predictions'] = np.floor(data['total_count_predictions']).astype(int)
            data = data.drop(columns=['campaign_id', 'at_least_one', 'at_least_two', 'at_least_three'])


            st.write('Итоговая таблица предсказанных просмотров пользователей в зависимости от выбранной аудитории')
            st.dataframe(data, use_container_width=True)

            # Преобразуем предсказания в файл для скачивания
            data.to_csv(output, sep='\t', index=False)
            output.seek(0)  # Возвращаем курсор в начало

            output_file_count = output.getvalue().encode('utf-8')
            # Скачивание результатов
            st.write("Предсказания успешно получены!")
            st.download_button('Скачать файл с предсказаниями (в долях)', output_file, file_name="predictions.tsv")
            st.download_button('Скачать файл с предсказаниями (в людях)', output_file_count, file_name="predictions_audience.tsv")

# Запуск приложения
if __name__ == "__main__":
    main()
