import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import OneHotEncoder
import os
import shutil


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
    print('Into function "extract_feature":')
    print('Extracting features...')
    df = load_data(data_path)
    print("Sorting lines by ['TERMINALNO', 'TIME']...")
    df.sort_values(by=['TERMINALNO', 'TIME'], inplace=True)
    features = []
    process_params = {}

    # >>>[1] Time features
    df['month'] = df['TIME'].apply(get_time, args=('month',))
    df['day'] = df['TIME'].apply(get_time, args=('day',))
    df['hour'] = df['TIME'].apply(get_time, args=('hour',))
    df['minute'] = df['TIME'].apply(get_time, args=('minute',))
    df['weekday'] = df['TIME'].apply(get_time, args=('weekday',))

    df['month'] = (df['month'] / 12.0).astype(np.float32)
    df['day'] = (df['day'] / 31.0).astype(np.float32)
    df['hour'] = ((df['hour'] + 1) / 24.0).astype(np.float32)
    df['minute'] = (df['minute'] / 60.0).astype(np.float32)
    df['weekday'] = ((df['weekday'] + 1) / 7.0).astype(np.float32)

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

    df['LONGITUDE'] = ((df['LONGITUDE'] - min_lon) / (max_lon - min_lon)).astype(np.float32)
    df['LATITUDE'] = ((df['LATITUDE'] - min_lat) / (max_lat - min_lat)).astype(np.float32)
    df['HEIGHT'] = ((df['HEIGHT'] - min_h) / (max_h - min_h)).astype(np.float32)

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

    df['SPEED'] = ((df['SPEED'] - mean_speed) / std_speed).astype(np.float32)

    features.append('SPEED')

    # >>>[4]Direction features
    df['DIRECTION'] = (df['DIRECTION'] / 360.0).astype(np.float32)

    features.append('DIRECTION')

    # >>>[5] Call state features
    if target is not None:
        ohe = OneHotEncoder()
        ohe_feat = ohe.fit_transform(df[['CALLSTATE']]).toarray().astype(np.uint8)
        process_params['ohe'] = ohe
    else:
        ohe = data_process_params['ohe']
        ohe_feat = ohe.transform(df[['CALLSTATE']]).toarray().astype(np.uint8)
    dummy_feat_names = ['CALLSTATE_'+str(i) for i in range(ohe_feat.shape[1])]
    dummy_feat = pd.DataFrame(ohe_feat, index=df.index, columns=dummy_feat_names)
    df = df.join(dummy_feat)

    features.extend(dummy_feat_names)
    print('Quit from function "extract_feature"')

    if target is None:
        return df[['TERMINALNO']+features], features, None
    else:
        return df[['TERMINALNO']+features + [target]], features, process_params


def save_data(df, feature_names, data_dir, target=None):

    length = []

    users = df['TERMINALNO'].unique()
    # make dir to save datas
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)

    os.chdir(data_dir)
    features_dir = os.path.join(data_dir, 'datas/')
    os.makedirs('datas/')

    # save a feature file for each user
    id_target = np.zeros((len(users), 2), dtype=np.float32)
    for idx, uid in enumerate(users):
        user_features = df.loc[df['TERMINALNO'] == uid, feature_names]
        length.append(user_features.shape[0])
        file_name = os.path.join(features_dir, str(idx)+r'.npy')
        np.save(file_name, user_features)
        id_target[idx, 0] = uid
        if target is not None:
            id_target[idx, 1] = df.loc[df['TERMINALNO'] == uid, target].values[0]

    # save the targes file
    targets_file_name = os.path.join(data_dir, 'targets.npy')
    np.save(targets_file_name, id_target)
    return length


def prepare_data(raw_data_path, target_data_dir, process_params=None, target=None):
    df_feat, feature_names, params = extract_feature(raw_data_path, process_params, target=target)
    lens = save_data(df_feat, feature_names, target_data_dir, target=target)
    return params, len(feature_names), lens


def train_test_split(data_path, test_ratio=0.25, random_state=0):

    feature_path = os.path.join(data_path, 'datas')
    data_files = np.array([os.path.join(feature_path, df) for df in os.listdir(feature_path)])
    data_files.sort()
    np.random.seed(random_state)
    np.random.shuffle(data_files)
    k = int(test_ratio*len(data_files))
    train = data_files[k:]
    test = data_files[:k]
    return train, test


def generate_xy(data_files, target_file, x_dim, batch_size=128, max_len=128, x_num=1):

    targets = np.load(target_file)
    if len(data_files) < batch_size:
        batches = 1
        batch_size = len(data_files)
    else:
        batches = len(data_files)//batch_size
    data_files = data_files.copy()
    while True:
        np.random.shuffle(data_files)
        for batch in range(batches):
            x = np.zeros((batch_size, max_len, x_dim), dtype=np.float32)
            y = []
            for idx, file_name in enumerate(data_files[batch:batch+batch_size]):
                user_idx = int(os.path.split(file_name)[1].split(r'.')[0])
                x_values = np.load(file_name)
                x_len = x_values.shape[0]
                # padding
                if x_len < max_len:
                    pad_len = max_len - x_len
                    pad_values = np.zeros((pad_len, x_dim))
                    x_values = np.concatenate([x_values, pad_values])
                # truncating
                if x_len > max_len:
                    trunc_len = x_len - max_len
                    k = np.random.choice(trunc_len)
                    x_values = x_values[k:x_len-trunc_len+k, :]
                x[idx, :, :] = x_values
                prob = 2.0 / (1+np.exp(-targets[user_idx, 1]))-1
                y.append(prob)
            if x_num > 1:
                x = [x]*x_num
            yield x, np.array(y)


def generate_x(data_files, x_dim, batch_size=128, max_len=128, x_num=1):

    data_len = len(data_files)
    if data_len < batch_size:
        batches = 1
    elif (data_len % batch_size) > 0:
        batches = data_len//batch_size+1
    else:
        batches = data_len//batch_size
    while True:
        for batch in range(batches):
            x = np.zeros((batch_size, max_len, x_dim), dtype=np.float32)
            for idx, file_name in enumerate(data_files[batch:min(batch+batch_size, data_len)]):
                user_idx = int(os.path.split(file_name)[1].split(r'.')[0])
                x_values = np.load(file_name)
                # padding
                if x_values.shape[0] < max_len:
                    pad_len = max_len - x_values.shape[0]
                    pad_values = np.zeros((pad_len, x_dim))
                    x_values = np.concatenate([x_values, pad_values])
                # truncating
                if x_values.shape[0] > max_len:
                    trunc_len = x_values.shape[0] - max_len
                    x_values = x_values[trunc_len:, :]
                x[idx, :, :] = x_values
            if x_num > 1:
                x = [x]*x_num
            yield x

