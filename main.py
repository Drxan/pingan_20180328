# -*- coding:utf-8 -*-
from pingan import data_helper, models
from pingan.data_helper import generate_xy, generate_x
from keras import losses
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import os

# ---------submit------------

path_train = '/data/dm/train.csv'
path_test = '/data/dm/test.csv'
path_test_out = "model/"  
'''
# --------local test---------
path_train = '/home/yw/study/Competition/pingan/train.csv'  # 训练文件
path_test = '/home/yw/study/Competition/pingan/test.csv'  # 测试文件
path_test_out = "model/"
'''

CURRENT_PATH = os.getcwd()
BATCH_SIZE = 128
EPOCHES = 1000


def process():

    print('>>>[1].Preprocessing train data...')
    train_data_path = os.path.join(CURRENT_PATH, 'data/train')

    params, feature_num, lens = data_helper.prepare_data(path_train, train_data_path, target='Y')
    os.chdir(CURRENT_PATH)

    print('>>>[2].Preprocessing test data...')
    test_data_path = os.path.join(CURRENT_PATH, 'data/test')
    _ = data_helper.prepare_data(path_test, test_data_path, process_params=params, target=None)
    os.chdir(CURRENT_PATH)

    print('>>>[3].Split data into the train and validate...')
    train_data, val_data = data_helper.train_test_split(train_data_path, test_ratio=0.2, random_state=9)
    target_file = os.path.join(train_data_path, 'targets.npy')
    max_len = int(np.percentile(lens, 80))
    x_dim = feature_num

    print('>>>[4].Creating model...')
    model = models.create_cnn((max_len, x_dim))
    model.compile(optimizer='adam', loss=losses.mse)
    print(model.summary())

    print('val steps:', len(val_data)//BATCH_SIZE)
    print('>>>[5].Training model...')
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    val_batch_size = int(len(val_data)/10)
    val_steps = len(val_data)//val_batch_size
    hist = model.fit_generator(generate_xy(train_data, target_file, x_dim, batch_size=BATCH_SIZE, max_len=max_len),
                               steps_per_epoch=max(len(train_data)//BATCH_SIZE, 1),
                               epochs=EPOCHES,
                               callbacks=[early_stop],
                               validation_data=generate_xy(val_data, target_file, x_dim, batch_size=val_batch_size, max_len=max_len),
                               validation_steps=val_steps,
                               initial_epoch=0)

    print('>>>[6].Predicting...')
    pred_batch_size = 256
    id_preds = np.load(os.path.join(test_data_path, 'targets.npy'))
    test_data, _ = data_helper.train_test_split(test_data_path, test_ratio=0)
    test_data_len = len(test_data)
    if test_data_len < pred_batch_size:
        pred_steps = 1
    elif (test_data_len % pred_batch_size) > 0:
        pred_steps = test_data_len // pred_batch_size + 1
    else:
        pred_steps = test_data_len // pred_batch_size
    predicts = model.predict_generator(generate_x(test_data, x_dim=x_dim, batch_size=pred_batch_size, max_len=max_len), steps=pred_steps)

    print('>>>[7].Saving results...')
    predicts = np.array(predicts).reshape(-1)
    for idx, file_name in enumerate(test_data):
        user_idx = int(os.path.split(file_name)[1].split(r'.')[0])
        id_preds[user_idx, 1] = predicts[idx]

    pred_csv = pd.DataFrame(columns=['Id', 'Pred'])
    pred_csv['Id'] = id_preds[:, 0].astype(np.int64)
    pred_csv['Pred'] = id_preds[:, 1]
    pred_csv.to_csv(path_test_out+'pred.csv', index=False)


if __name__ == "__main__":
    print("****************** start **********************")
    process()
