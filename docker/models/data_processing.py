import pandas as pd
import lightgbm as lgb


def preprocessing(history, users, validate):
    # Вычисляем границы выбросов
    Q1 = history['cpm'].quantile(0.25)
    Q3 = history['cpm'].quantile(0.75)
    IQR = Q3 - Q1
    history = history[(history['cpm'] >= Q1 - 1.5 * IQR) & (history['cpm'] <= Q3 + 1.5 * IQR)]
        
    # Подсчет частоты встречаемости площадок
    publisher_counts = history['publisher'].value_counts(normalize=True)  # Доля вхождений

    # Применяем Frequency Encoding
    history['publisher_freq'] = history['publisher'].map(publisher_counts)

    # Обрабатываем validate (если есть новые площадки, заменяем их на 0)
    validate_publishers = validate['publishers'].str.split(',')
    validate_freqs = validate_publishers.apply(lambda pubs: [publisher_counts.get(int(p), 0) for p in pubs])
    validate['publisher_freq_mean'] = validate_freqs.apply(lambda x: sum(x) / len(x) if x else 0)
    
    # Фичи по пользователям
    user_stats = history.groupby('user_id').agg(
        user_total_ads=('hour', 'count'),  # Общее число показов
        user_mean_cpm=('cpm', 'mean'),  # Средний CPM
        user_max_cpm=('cpm', 'max'),  # Максимальный CPM
        user_unique_publishers=('publisher', 'nunique'),  # Кол-во уникальных площадок
    ).reset_index()

    # Фичи по времени (часу)
    hour_stats = history.groupby('hour').agg(
        hour_total_ads=('user_id', 'count'),
        hour_mean_cpm=('cpm', 'mean')
    ).reset_index()


    # Пройдем по всем строкам в validate
    for i, row in validate.iterrows():
        # Извлекаем время начала и окончания
        start_time = row['hour_start']
        end_time = row['hour_end']
        
        # Фильтруем часовые данные, которые попадают в интервал
        hour_subset = hour_stats[(hour_stats['hour'] >= start_time) & (hour_stats['hour'] <= end_time)]
        
        # Агрегируем статистику для выбранных часов
        total_ads_in_period = hour_subset['hour_total_ads'].sum()
        mean_cpm_in_period = hour_subset['hour_mean_cpm'].mean()
        
        # Добавляем эти фичи в текущую строку датасета
        validate.at[i, 'total_ads_in_period'] = total_ads_in_period
        validate.at[i, 'mean_cpm_in_period'] = mean_cpm_in_period

    # Теперь в датасете validate есть фичи 'total_ads_in_period' и 'mean_cpm_in_period'

    # Приведение списка user_ids в validate.tsv к удобному формату
    validate['user_ids'] = validate['user_ids'].apply(lambda x: list(map(int, x.split(','))))

    validate['start_hour_of_day'] = validate['hour_start'] % 24
    validate['end_hour_of_day'] = validate['hour_end'] % 24
    validate["duration"] = (validate["hour_end"] - validate["hour_start"]).clip(0, 24)


    # Подсчет количества показов рекламы для каждого пользователя
    user_exposure_counts = history.groupby('user_id').size().reset_index(name='num_exposures')

    # Объединение с демографическими данными пользователей
    user_features = users.merge(user_exposure_counts, on='user_id', how='left').fillna(0)

    user_features['max_user_cpm'] = history.groupby('user_id')['cpm'].transform('max')

    # Объединение статистики по пользователям с демографическими данными
    user_features = user_features.merge(user_stats, on='user_id', how='left')

    # Создание обучающего датасета
    train_data = []
    for i, row in validate.iterrows():
        user_subset = user_features[user_features['user_id'].isin(row['user_ids'])].copy()
        user_subset['campaign_id'] = i  # идентификатор рекламной кампании
        user_subset['audience_size'] = row['audience_size']
        user_subset['cpm'] = row['cpm']
        user_subset['hour_start'] = row['hour_start']
        user_subset['hour_end'] = row['hour_end']
        user_subset['publisher_freq_mean'] = row['publisher_freq_mean']
        
        # Добавление фич по времени
        user_subset['start_hour_of_day'] = row['hour_start'] % 24
        user_subset['end_hour_of_day'] = row['hour_end'] % 24
        user_subset["duration"] = (user_subset["hour_end"] - user_subset["hour_start"]).clip(0, 24)

        # Статистика по пользователю
        user_subset['user_total_ads'] = user_subset['user_id'].map(user_stats.set_index('user_id')['user_total_ads']).fillna(0)
        user_subset['user_mean_cpm'] = user_subset['user_id'].map(user_stats.set_index('user_id')['user_mean_cpm']).fillna(0)
        user_subset['user_max_cpm'] = user_subset['user_id'].map(user_stats.set_index('user_id')['user_max_cpm']).fillna(0)
        user_subset['user_unique_publishers'] = user_subset['user_id'].map(user_stats.set_index('user_id')['user_unique_publishers']).fillna(0)

        # Добавление статистики по времени (hour_stats) для интервала от hour_start до hour_end
        # Фильтруем по часам, которые находятся в интервале между hour_start и hour_end
        hour_subset = hour_stats[(hour_stats['hour'] >= row['hour_start']) & (hour_stats['hour'] <= row['hour_end'])]
        user_subset['total_ads_in_period'] = hour_subset['hour_total_ads'].sum()  # Суммируем total_ads
        user_subset['mean_cpm_in_period'] = hour_subset['hour_mean_cpm'].mean()  # Среднее значение CPM
        
        train_data.append(user_subset)

    # Объединяем все в один DataFrame
    train_df = pd.concat(train_data, ignore_index=True)


    return train_df


def generate_predictions(train_df):

    model_1 = lgb.Booster(model_file='models/model_1.txt')
    model_2 = lgb.Booster(model_file='models/model_2.txt')
    model_3 = lgb.Booster(model_file='models/model_3.txt')

    train_df_without_campaign = train_df.drop(columns=['campaign_id'])

    y_pred_1 = model_1.predict(train_df_without_campaign)
    y_pred_2 = model_2.predict(train_df_without_campaign)
    y_pred_3 = model_3.predict(train_df_without_campaign)

    train_df['at_least_one'] = y_pred_1
    train_df['at_least_two'] = y_pred_2
    train_df['at_least_three'] = y_pred_3

    campaign_grouped_predictions = train_df.groupby('campaign_id').agg({
    'at_least_one': 'mean',
    'at_least_two': 'mean',
    'at_least_three': 'mean'
    }).reset_index()

    return campaign_grouped_predictions








