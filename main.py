﻿# -*- coding:utf-8 -*-
from pingan import data_helper, models
from keras import losses
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd

# ---------submit------------
'''
path_train = '/data/dm/train.csv' # r'D:\yuwei\study\competition\pingan\train.csv'  # 训练文件
path_test = '/data/dm/test.csv' # r'D:\yuwei\study\competition\pingan\test.csv'  # 测试文件
path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。
'''
# --------local test---------
path_train = r'D:\yuwei\study\competition\pingan\train.csv'  # 训练文件
path_test = r'D:\yuwei\study\competition\pingan\test.csv'  # 测试文件
path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。


def process():
    print('>>>(1).Preparing train data...')
    x_train, y_train, process_params, _ = data_helper.get_xy(path_train, target='Y')
    print(x_train.shape, x_train[0].shape)

    print('>>>(2).Creating model...')
    model = models.create_model(x_train[0].shape, drop_out=0.5)
    model.compile(optimizer='adam', loss=losses.mse)
    print(model.summary())

    print('>>>(3).Training model...')
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    hist = model.fit(x_train, y_train, batch_size=128, epochs=1000, validation_split=0.2, callbacks=[early_stop])

    del x_train
    del y_train

    print('>>>(4).Preparing test data...')
    x_test, _, _, users = data_helper.get_xy(path_test, process_params)

    print('>>>(5).Predicting...')
    predicts = model.predict(x_test)

    print('>>>(6).Saving results...')
    predicts = np.array(predicts).reshape(-1)
    pred_csv = pd.DataFrame(columns=['Id', 'Pred'])
    pred_csv['Id'] = users
    pred_csv['Pred'] = predicts
    pred_csv.to_csv(path_test_out+'pred.csv', index=False)


if __name__ == "__main__":
    print("****************** start **********************")
    process()
