# -*- coding:utf-8 -*-
import xgboost as xgb
import numpy as np
import pandas as pd


# load the xgboost model
def load_model(model_path):
    model_load = xgb.Booster(model_file=model_path)
    return model_load


# load the libsvm_data
def load_libsvm_data(libsvm_data_path):
    data_load = xgb.DMatrix(libsvm_data_path)
    return data_load


# load the dense data
# you can get dense_data and label_data from libsvm_data using libsvm2dense.py
def load_dense_data(dense_data_path, data_form='numpy_array'):
    feature_list = list()
    label_list = list()
    with open(dense_data_path, 'r') as dense_f:
        for line in dense_f:
            # xgboost训练和预测使用的数据格式必须保持一致,否则预测结果会异常
            # 比如model下的LocalFile模型是用libsvm训练的，预测时候使用的数据必须使用libsvm格式
            # 但我这里想用numpy array或者pandas dataframe，则只需在特征列前加一维度即可，可以是任意数字
            temp_list = [-1.0]
            line_list = line.strip().split(' ')
            feature_list.append(temp_list + line_list)
    if data_form == 'pandas_dataframe':
        data_load = xgb.DMatrix(pd.DataFrame(feature_list), label=label_list)
    else:
        data_load = xgb.DMatrix(np.array(feature_list), label=label_list)
    return data_load
