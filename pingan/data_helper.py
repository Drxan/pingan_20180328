import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import OneHotEncoder


def load_data(data_path):
    df = pd.read_csv(data_path)
    return df


def get_time(unix_time_stamp, time_type='hour'):
    time_value = None
    try:
        dtime = datetime.datetime.fromtimestamp(unix_time_stamp)
        if time_type == 'year':
            time_value = dtime.year
        if time_type == 'month':
            time_value = dtime.month
        if time_type == 'day':
            time_value = dtime.day
        if time_type == 'hour':
            time_value = dtime.hour
        if time_type == 'minute':
            time_value = dtime.minute + dtime.second/60.0
        if time_type == 'weekday':
            time_value = dtime.weekday()
    except Exception as e:
        print("Error:", e)
    return time_value


def extract_feature(data_path, data_process_params=None, target=None):
    df = load_data(data_path)
    df.sort_values(by=['TERMINALNO', 'TIME'], inplace=True)
    features = []
    process_params = {}

    # >>>[1] Time features
    df['month'] = df['TIME'].apply(get_time, args=('month',))
    df['day'] = df['TIME'].apply(get_time, args=('day',))
    df['hour'] = df['TIME'].apply(get_time, args=('hour',))
    df['minute'] = df['TIME'].apply(get_time, args=('minute',))
    df['weekday'] = df['TIME'].apply(get_time, args=('weekday',))

    df['month'] = df['month'] / 12.0
    df['day'] = df['day'] / 31.0
    df['hour'] = (df['hour'] + 1) / 24.0
    df['minute'] = df['minute'] / 60.0
    df['weekday'] = (df['weekday'] + 1) / 7.0

    features.extend(['month', 'day', 'hour', 'minute', 'weekday'])

    # >>>[2] Location features
    if target is not None:
        max_lon = df['LONGITUDE'].max()
        min_lon = df['LONGITUDE'].min()
        max_lat = df['LATITUDE'].max()
        min_lat = df['LATITUDE'].min()
        max_h = df['HEIGHT'].max()
        min_h = df['HEIGHT'].min()

        process_params['max_lon'] = max_lon
        process_params['min_lon'] = min_lon
        process_params['max_lat'] = max_lat
        process_params['min_lat'] = min_lat
        process_params['max_h'] = max_h
        process_params['min_h'] = min_h
    else:
        max_lon = data_process_params['max_lon']
        min_lon = data_process_params['min_lon']
        max_lat = data_process_params['max_lat']
        min_lat = data_process_params['min_lat']
        max_h = data_process_params['max_h']
        min_h = data_process_params['min_h']

    df['LONGITUDE'] = (df['LONGITUDE'] - min_lon) / (max_lon - min_lon)
    df['LATITUDE'] = (df['LATITUDE'] - min_lat) / (max_lat - min_lat)
    df['HEIGHT'] = (df['HEIGHT'] - min_h) / (max_h - min_h)

    features.extend(['LONGITUDE', 'LATITUDE', 'HEIGHT'])

    # >>>[3]Speed features
    if target is not None:
        mean_speed = df['SPEED'].mean()
        std_speed = df['SPEED'].std()

        process_params['mean_speed'] = mean_speed
        process_params['std_speed'] = std_speed
    else:
        mean_speed = data_process_params['mean_speed']
        std_speed = data_process_params['std_speed']

    df['SPEED'] = (df['SPEED'] - mean_speed) / std_speed

    features.append('SPEED')

    # >>>[4]Direction features
    df['DIRECTION'] = df['DIRECTION'] / 360.0

    features.append('DIRECTION')

    # >>>[5] Call state features
    if target is not None:
        ohe = OneHotEncoder()
        ohe_feat = ohe.fit_transform(df[['CALLSTATE']]).toarray()
        process_params['ohe'] = ohe
    else:
        ohe = data_process_params['ohe']
        ohe_feat = ohe.transform(df[['CALLSTATE']]).toarray()
    dummy_feat_names = ['CALLSTATE_'+str(i) for i in range(ohe_feat.shape[1])]
    dummy_feat = pd.DataFrame(ohe_feat, index=df.index, columns=dummy_feat_names)
    df = df.join(dummy_feat)

    features.extend(dummy_feat_names)

    if target is None:
        return df[['TERMINALNO']+features], features, None
    else:
        return df[['TERMINALNO']+features + [target]], features, process_params


def prepare_model_data(feature_df, max_len, features, target=None):
    x_dim = len(features)
    users = feature_df['TERMINALNO'].unique()

    x_list = np.zeros((len(users), max_len, x_dim)).astype(np.float32)
    y_list = []

    if target:
        for idx, uid in enumerate(users):
            x_values = feature_df.loc[feature_df['TERMINALNO'] == uid, features]
            y = feature_df.loc[feature_df['TERMINALNO'] == uid, target].values[0]
            # padding
            if x_values.shape[0] < max_len:
                pad_len = max_len - x_values.shape[0]
                pad_values = np.zeros((pad_len, x_dim))
                x_values = np.concatenate([x_values, pad_values])
            # truncating
            if x_values.shape[0] > max_len:
                trunc_len = x_values.shape[0] - max_len
                x_values = x_values.values[trunc_len:, :]
            x_list[idx, :, :] = x_values.astype(np.float32)
            y_list.append(y)
    else:
        for idx, uid in enumerate(users):
            x_values = feature_df.loc[feature_df['TERMINALNO'] == uid, features]
            # padding
            if x_values.shape[0] < max_len:
                pad_len = max_len - x_values.shape[0]
                pad_values = np.zeros((pad_len, x_dim))
                x_values = np.concatenate([x_values, pad_values])
            # truncating
            if x_values.shape[0] > max_len:
                trunc_len = x_values.shape[0] - max_len
                x_values = x_values.values[trunc_len:, :]
            x_list[idx, :, :] = x_values.astype(np.float32)
    return x_list, np.array(y_list).astype(np.float32), users


def get_xy(data_path, data_process_params=None, target=None, max_len=None):
    feature_df, features, process_params = extract_feature(data_path, data_process_params, target)
    if max_len is None:
        max_len = int(feature_df['TERMINALNO'].value_counts().mean())
    x_data, y_data, users= prepare_model_data(feature_df, max_len, features, target=target)
    return x_data, y_data, process_params, users

