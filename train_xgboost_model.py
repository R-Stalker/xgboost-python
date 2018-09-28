# -*- coding: utf-8 -*-

import xgboost as xgb
import my_util

if __name__ == '__main__':
    train_data_path = 'train_data'
    pre_data_path = 'pre_data'
    train_data = my_util.load_libsvm_data(train_data_path)
    eval_data = my_util.load_libsvm_data(pre_data_path)

    param = [('max_depth', 4),
             ('booster', 'gbtree'),
             ('objective', 'binary:logistic'),
             ('eval_metric', 'error'),
             ('eval_metric', 'logloss'),
             ('eval_metric', 'auc'),
             ('eta', 0.1),
             ('subsample', 0.8),
             ('colsample_bytree', 0.8),
             ('min_child_weight', 6),
             ('scale_pos_weight', 6),
             ('gamma', 0.1),
             ('lambda', 2),
             ('alpha', 1)
             ]

    num_round = 450
    watchlist = [(eval_data, 'eval'), (train_data, 'train')]

    evals_result = {}
    bst = xgb.train(param, train_data, num_round, watchlist, evals_result=evals_result)

    bst.save_model("output_file\\test_model")

    print('Access logloss metric directly from evals_result:')
    print(evals_result['eval']['logloss'])

    print('Access metrics through a loop:')
    for e_name, e_mtrs in evals_result.items():
        print('- {}'.format(e_name))
        for e_mtr_name, e_mtr_vals in e_mtrs.items():
            print('   - {}'.format(e_mtr_name))
            print('      - {}'.format(e_mtr_vals))

    print('Access complete dictionary:')
    print(evals_result)
