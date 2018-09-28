# -*- coding:utf-8 -*-
import my_util

if __name__ == '__main__':
    tesla_model_path = 'model\\LocalFile'
    predict_data_path = 'output_file\\predict.result'

    # load model
    my_model = my_util.load_model(tesla_model_path)

    # load data for predict
    test_data = my_util.load_dense_data('dense.features', 'dense.labels')
    # test_data = my_util.load_libsvm_data('pre_data')

    # predict
    predict_data = my_model.predict(test_data)
    print ('There are ' + str(len(predict_data)) + ' rows of data.')

    # save the predict result
    with open(predict_data_path, "w") as pre_f:
        for i in range(len(predict_data)):
            pre_f.write(str(predict_data[i]) + '\n')
    print("Prediction result has saved in " + str(predict_data_path))
