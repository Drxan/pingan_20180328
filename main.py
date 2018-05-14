# -*- coding:utf-8 -*-
from pingan import data_helper_mulprocess, models
from pingan.data_helper_mulprocess import generate_xy, generate_x
from keras import losses
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import os
import time
from keras import metrics
from pingan import submit_model
# ---------submit------------
'''
path_train = '/data/dm/train.csv'
path_test = '/data/dm/test.csv'
path_test_out = "model/"  
'''
# --------local test---------
path_train = '/home/yw/study/Competition/pingan/train.csv'  # 训练文件
path_test = '/home/yw/study/Competition/pingan/test.csv'  # 测试文件
path_test_out = "model/"


CURRENT_PATH = os.getcwd()


if __name__ == "__main__":
    print("****************** start **********************")
    submit_model.dnn_old.process(CURRENT_PATH)
