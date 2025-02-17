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

def get_smoothed_mean_log_accuracy_ratio(answers, responses, epsilon=0.005):
    log_accuracy_ratio_mean = np.array(
        [
            get_smoothed_log_mape_column_value(responses['at_least_one'], answers['at_least_one'], epsilon),
            get_smoothed_log_mape_column_value(responses['at_least_two'], answers['at_least_two'], epsilon),
            get_smoothed_log_mape_column_value(responses['at_least_three'], answers['at_least_three'], epsilon),
        ]
    ).mean()

    percentage_error = 100 * (np.exp(log_accuracy_ratio_mean) - 1)

    return percentage_error.round(
        decimals=2
    )

def main():
    answers_filename = sys.argv[1]
    responses_filename = sys.argv[2]

    answers = load_answers(answers_filename)
    responses = load_answers(responses_filename)

    print(get_smoothed_mean_log_accuracy_ratio(answers, responses))



if __name__ == '__main__':
    main()
