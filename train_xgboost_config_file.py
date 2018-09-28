# -*- coding:utf-8 -*-

# 通过读取配置文件的方式进行模型训练，配置文件示例是train_xgboost.conf

import os
import my_util
import ConfigParser
import xgboost as xgb


# 通过读取配置文件来获取模型训练参数，当然也可以不使用这个方法,参见train_xgboost_model.py
def get_model_params(config):
    booster = config.get("xgboost_params", "booster")
    objective = config.get("xgboost_params", "objective")
    eta = float(config.get("xgboost_params", "eta"))
    max_depth = int(config.get("xgboost_params", "max_depth"))
    subsample = float(config.get("xgboost_params", "subsample"))
    min_child_weight = int(config.get("xgboost_params", "min_child_weight"))
    col_sample_bytree = float(config.get("xgboost_params", "colsample_bytree"))
    scale_pos_weight = int(config.get("xgboost_params", "scale_pos_weight"))
    eval_metric = config.get("xgboost_params", "eval_metric")
    gamma = float(config.get("xgboost_params", "gamma"))
    l1_alpha = int(config.get("xgboost_params", "alpha"))
    l2_lambda = int(config.get("xgboost_params", "lambda"))
    silent = int(config.get("xgboost_params", "silent"))
    threshold = float(config.get("xgboost_params", "threshold"))
    num_boost_round = int(config.get("xgboost_params", "num_boost_round"))

    print("------------xgboost params-----------------")
    print("booster = %s" % str(booster))
    print("objective = %s" % str(objective))
    print("eta = %s" % str(eta))
    print("max_depth = %s" % str(max_depth))
    print("subsample = %s" % str(subsample))
    print("min_child_weight = %s" % str(min_child_weight))
    print("col_sample_bytree = %s" % str(col_sample_bytree))
    print("scale_pos_weight = %s" % str(scale_pos_weight))
    print("eval_metric = %s" % str(eval_metric))
    print("gamma = %s" % str(gamma))
    print("l1_alpha = %s" % str(l1_alpha))
    print("l2_lambda = %s" % str(l2_lambda))
    print("silent = %s" % str(silent))
    print("threshold = %s" % str(threshold))
    print("num_boost_round = %s" % str(num_boost_round))

    params = dict()
    params['booster'] = booster
    params['objective'] = objective
    params['eta'] = eta
    params['max_depth'] = max_depth
    params['subsample'] = subsample
    params['min_child_weight'] = min_child_weight
    params['colsample_bytree'] = col_sample_bytree
    params['scale_pos_weight'] = scale_pos_weight
    params['eval_metric'] = eval_metric
    params['gamma'] = gamma
    params['alpha'] = l1_alpha
    params['lambda'] = l2_lambda
    params['silent'] = silent
    params['threshold'] = threshold
    return params


# train and save the model
def train_model(config_path):
    conf = ConfigParser.ConfigParser()
    conf.read(config_path)

    train_data_path = str(conf.get("path", "train_data_path"))
    test_data_path = str(conf.get("path", "test_data_path"))
    save_model_path = str(conf.get("path", "save_model_path"))
    print("train_data_path = %s" % train_data_path)
    print("test_data_path = %s" % test_data_path)
    print("save_model_path = %s" % save_model_path)

    params = get_model_params(conf)

    train_data = my_util.load_libsvm_data(train_data_path)
    test_data = my_util.load_libsvm_data(test_data_path)
    watchlist = [(train_data, 'train'), (test_data, 'eval')]
    num_boost_round = int(conf.get("xgboost_params", "num_boost_round"))

    model_train = xgb.train(params, train_data, num_boost_round, watchlist)
    model_train.save_model(save_model_path)


if __name__ == '__main__':
    config_file_path = 'train_xgboost.conf'
    train_model(config_file_path)
