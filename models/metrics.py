import sys
import pandas as pd
import numpy as np


def load_answers(answers_filename):
    return pd.read_csv(answers_filename, sep=",")  # замените на корректный разделитель


def get_smoothed_log_mape_column_value(responses_column, answers_column, epsilon):
    return np.abs(np.log(
        (responses_column + epsilon)
        / (answers_column + epsilon)
    )).mean()


def get_smoothed_log_accuracy_ratio(answers, responses, epsilon=0.005):
    log_accuracy_ratio_dict = {}

    # Перебираем все столбцы, начиная с 'at_least_one', 'at_least_two', 'at_least_three' и т.д.
    for column in ['at_least_one', 'at_least_two', 'at_least_three']:
        log_accuracy_ratio_dict[column] = get_smoothed_log_mape_column_value(
            responses[column], answers[column], epsilon
        )

    # Рассчитываем процентную ошибку для каждого столбца
    percentage_errors = {
        column: 100 * (np.exp(value) - 1) for column, value in log_accuracy_ratio_dict.items()
    }

    return percentage_errors


def main():
    answers_filename = sys.argv[1]
    responses_filename = sys.argv[2]

    answers = load_answers(answers_filename)
    responses = load_answers(responses_filename)

    # Получаем ошибку для каждого столбца
    percentage_errors = get_smoothed_log_accuracy_ratio(answers, responses)

    # Выводим ошибку для каждого столбца
    for column, error in percentage_errors.items():
        print(f"Ошибка для {column}: {error:.2f}%")


if __name__ == '__main__':
    main()
