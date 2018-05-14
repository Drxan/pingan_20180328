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


CURRENT_PATH = os.getcwd()


if __name__ == "__main__":
    print("****************** start **********************")
    submit_model.M20180515_04LightGBM.process(CURRENT_PATH)
