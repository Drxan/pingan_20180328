import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import OneHotEncoder
import os
import shutil
import math


def extract_feature(raw_data_path, dtype, save_path, data_process_params=None, target=None):
    # make dir to save datas
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    os.chdir(save_path)
    features_dir = os.path.join(save_path, 'datas/')
    os.makedirs('datas/')

    process_params = {}

    df = pd.read_csv(raw_data_path, dtype=dtype)
    # (1) 过滤掉方向未知或速度未知的记录，同时过滤掉高度小于-50m的记录
    df = df.loc[(df['DIRECTION'] >= 0) & (df['SPEED'] >= 0) & (df['HEIGHT'] >= -50)].copy()

    # (2) normalize features
    if target is not None:
        mean_lon = df['LONGITUDE'].mean()
        std_lon = df['LONGITUDE'].std()
        mean_lat = df['LATITUDE'].mean()
        std_lat = df['LATITUDE'].std()
        mean_h = df['HEIGHT'].mean()
        std_h = df['HEIGHT'].std()

        process_params['mean_lon'] = mean_lon
        process_params['std_lon'] = std_lon
        process_params['mean_lat'] = mean_lat
        process_params['std_lat'] = std_lat
        process_params['mean_h'] = mean_h
        process_params['std_h'] = std_h
    else:
        mean_lon = data_process_params['mean_lon']
        std_lon = data_process_params['std_lon']
        mean_lat = data_process_params['mean_lat']
        std_lat = data_process_params['std_lat']
        mean_h = data_process_params['mean_h']
        std_h = data_process_params['std_h']
    df['LONGITUDE'] = ((df['LONGITUDE'] - mean_lon) / std_lon).astype(np.float32)
    df['LATITUDE'] = ((df['LATITUDE'] - mean_lat) / std_lat).astype(np.float32)
    df['HEIGHT'] = ((df['HEIGHT'] - mean_h) / std_h).astype(np.float32)

    if target is not None:
        mean_speed = df['SPEED'].mean()
        std_speed = df['SPEED'].std()

        process_params['mean_speed'] = mean_speed
        process_params['std_speed'] = std_speed
    else:
        mean_speed = data_process_params['mean_speed']
        std_speed = data_process_params['std_speed']
    df['SPEED'] = ((df['SPEED'] - mean_speed) / std_speed).astype(np.float32)

    df['DIRECTION'] = (df['DIRECTION'] / 360.0).astype(np.float32)

    # (3) start to extract features
    users = df['TERMINALNO'].unique()
    ufeatures = []
    id_target = np.zeros((len(users), 2), dtype=np.float32)
    lens = []
    for idx, uid in enumerate(users):
        udf = df.loc[df['TERMINALNO'] == uid]
        ufeature, utrip_features = extract_user_feature(udf)
        lens.append(utrip_features.shape[0])
        # save trip-level features
        file_name = os.path.join(features_dir, str(idx) + r'.npy')
        np.save(file_name, utrip_features)

        ufeatures.append(ufeature)
        id_target[idx, 0] = uid
        if target is not None:
            id_target[idx, 1] = udf[target].values[0]
    # save the userf features and targes file
    user_feature_file_name = os.path.join(save_path, 'ufeatures.npy')
    np.save(user_feature_file_name, np.array(ufeatures))
    targets_file_name = os.path.join(save_path, 'targets.npy')
    np.save(targets_file_name, id_target)

    if target is None:
        return ufeatures, lens, None
    else:
        return ufeatures, lens, process_params


def get_time(unix_time_stamp):
    dtime = datetime.datetime.fromtimestamp(unix_time_stamp)
    hour = dtime.hour
    weekday = dtime.weekday()+1
    return pd.Series([hour, weekday], index=['hour', 'weekday'])


def extract_user_feature(udf):
    # 过滤掉重复记录
    udf = udf.drop_duplicates()
    # 获得当前用户所有的trip id
    trips = udf['TRIP_ID'].unique()
    # 用于存放每个trip的特征向量
    trip_feature_names = ['month', 'day', 'hour', 'weekday'] + ['min_lat', 'max_lat', 'min_lon', 'max_lon'] + [
        'total_hdiff', 'max_hdiff', 'min_hdiff', 'mean_hdiff', 'std_hdiff', 'mean_height'] + ['total_direc_diff',
                                                                                              'max_direc_diff',
                                                                                              'min_direc_diff',
                                                                                              'mean_direc_diff',
                                                                                              'std_direc_diff'] + [
                             'total_time', 'total_length'] + ['max_speed', 'min_speed', 'mean_speed', 'std_speed'] + [
                             'max_point_speed', 'min_point_speed', 'mean_point_speed', 'std_point_speed']
    trip_features = []
    # 针对当前用户的每一条trip，提取行程级的特征向量，每个行程对应一个特征向量
    for trip in trips:
        trip_df = udf.loc[udf['TRIP_ID'] == trip]
        # 只对节点数大于1的trip进行特征抽取
        if trip_df.shape[0] >= 2:
            trip_feature = extract_trip_feature(trip_df)
            if trip_feature is not None:
                trip_features.append(trip_feature)
    # 如果该用户的所有行程都无法提取到有效特征，则添加一个各分量皆为零的虚拟行程的特征向量
    if len(trip_features) == 0:
        trip_features.append([0] * len(trip_feature_names))
    trip_features = pd.DataFrame(trip_features, columns=trip_feature_names)

    # 提取该用户的用户级特征，每个用户一个特征向量
    user_feature_names = []
    user_feature = []
    # 该用户有效速度特征
    real_speeds = udf.loc[udf['SPEED'] > 0, 'SPEED']
    user_feature.append(real_speeds.max())
    user_feature.append(real_speeds.min())
    user_feature.append(real_speeds.mean())
    user_feature.append(real_speeds.std())
    user_feature_names.extend(['max_point_speed', 'min_point_speed', 'mean_point_speed', 'std_point_speed'])

    # 该用户驾驶时间特征
    utime = udf['TIME'].apply(get_time)
    user_feature.append(utime['hour'].value_counts().iloc[0] / 23.0)
    user_feature.append(utime['weekday'].value_counts().iloc[0] / 7.0)
    user_feature.append(trip_features['hour'].value_counts().iloc[0])
    user_feature_names.extend(['most_hour', 'most_weekday', 'most_start_hour'])

    # 用户的位置几何中心
    loca = udf[['LONGITUDE', 'LATITUDE', 'HEIGHT']].mean()
    user_feature.append(loca['LONGITUDE'])
    user_feature.append(loca['LATITUDE'])
    user_feature.append(loca['HEIGHT'])
    user_feature_names.extend(['central_lon', 'central_lat', 'central_height'])

    # 用户行驶中常见通话状态占比
    calls = udf.loc[udf['SPEED'] > 0, 'CALLSTATE'].value_counts()
    calls = calls / calls.sum()
    call_rate = [0] * 5
    for i in range(len(calls)):
        call_rate[calls.index[i]] = calls.iloc[i]
    user_feature.extend(call_rate)
    user_feature_names.extend(['0_call', '1_call', '2_call', '3_call', '4_call'])

    return user_feature, trip_features


def extract_trip_feature(utrip):
    utrip = utrip.sort_values(by=['TIME'])
    all_speed_zero = (utrip['SPEED'] == 0).sum() == utrip.shape[0]
    if all_speed_zero:  # 如果该行程中每个节点的速度都为零，则认为该行程无效
        return None
    trip_feature = []
    feature_names = []
    speeds = []
    height_diffs = []
    direc_diffs = []
    times = 0
    dist = 0
    for i in range(utrip.shape[0] - 1):
        pre_speed = utrip['SPEED'].iloc[i]
        sub_speed = utrip['SPEED'].iloc[i + 1]
        if (pre_speed == 0) and (sub_speed == 0):
            continue
        # 两个节点间的时长
        time_dur = int(max(utrip['TIME'].iloc[i + 1] - utrip['TIME'].iloc[i], 60) / 60.0)
        times = times + time_dur
        # 节点之间的平均速度
        distance = utrip[['LONGITUDE', 'LATITUDE', 'HEIGHT']].iloc[i] - utrip[['LONGITUDE', 'LATITUDE', 'HEIGHT']].iloc[
            i + 1]
        distance = math.sqrt((distance ** 2).sum())
        dist = dist + distance
        mean_speed = distance / time_dur
        speeds.extend([mean_speed] * time_dur)
        # 节点之间的高差变化
        hdiff = (utrip['HEIGHT'].iloc[i + 1] - utrip['HEIGHT'].iloc[i]) / time_dur
        height_diffs.extend([hdiff] * time_dur)
        # 节点之间的方向变化（无正负之分）
        direc_diff = abs(utrip['DIRECTION'].iloc[i + 1] - utrip['DIRECTION'].iloc[i]) / time_dur
        direc_diffs.extend([direc_diff]*time_dur)

    # 当前行程的开始时间
    dtime = datetime.datetime.fromtimestamp(utrip['TIME'].iloc[0])
    trip_feature.append(dtime.month / 12.0)
    trip_feature.append(dtime.day / 31.0)
    trip_feature.append((dtime.hour + dtime.minute / 60 + dtime.second / 3600) / 24.0)
    trip_feature.append((dtime.weekday() + 1) / 7.0)
    # feature_names.extend(['month','day','hour','weekday'])

    # 当前行程的经纬度范围
    trip_feature.append(utrip['LATITUDE'].min())
    trip_feature.append(utrip['LATITUDE'].max())
    trip_feature.append(utrip['LONGITUDE'].min())
    trip_feature.append(utrip['LONGITUDE'].max())
    # feature_names.extend(['min_lat','max_lat','min_lon','max_lon'])

    # 当前行程中有效行驶的高差特征
    trip_feature.append(sum(height_diffs))
    trip_feature.append(max(height_diffs))
    trip_feature.append(min(height_diffs))
    trip_feature.append(np.mean(height_diffs))
    trip_feature.append(np.std(height_diffs))
    # 当前行程中有效行驶节点的高程特征
    trip_feature.append(utrip.loc[utrip['SPEED'] != 0, 'HEIGHT'].mean())
    # feature_names.extend(['total_hdiff','max_hdiff','min_hdiff','mean_hdiff','std_hdiff','mean_height'])

    # 当前行程中方向变化量特征
    trip_feature.append(sum(direc_diffs))
    trip_feature.append(max(direc_diffs))
    trip_feature.append(min(direc_diffs))
    trip_feature.append(np.mean(direc_diffs))
    trip_feature.append(np.std(direc_diffs))
    # feature_names.extend(['total_direc_diff','max_direc_diff','min_direc_diff','mean_direc_diff','std_direc_diff'])

    # 当前行程总的行程时长、总的行程长度
    trip_feature.append(times)
    trip_feature.append(dist)
    # feature_names.extend(['total_time','total_length'])

    # 当前行程的平均速度特征
    trip_feature.append(max(speeds))
    trip_feature.append(min(direc_diffs))
    trip_feature.append(np.mean(direc_diffs))
    trip_feature.append(np.std(direc_diffs))
    # feature_names.extend(['max_speed','min_speed','mean_speed','std_speed'])

    # 当前行程瞬时速度特征
    trip_feature.append(utrip.loc[utrip['SPEED'] > 0, 'SPEED'].max())
    trip_feature.append(utrip.loc[utrip['SPEED'] > 0, 'SPEED'].min())
    trip_feature.append(utrip.loc[utrip['SPEED'] > 0, 'SPEED'].mean())
    trip_feature.append(utrip.loc[utrip['SPEED'] > 0, 'SPEED'].std())
    # feature_names.extend(['max_point_speed','min_point_speed','mean_point_speed','std_point_speed'])
    return trip_feature