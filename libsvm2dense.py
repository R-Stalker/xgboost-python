# -*- coding:utf-8 -*-
import os


def handle_one_file(file_input_path, dense_data_list, label_list):
    with open(file_input_path, 'r') as f:
        for line in f:
            line_result = list()
            line_data_list = line.strip().split(' ')
            label_list.append(line_data_list[0])
            for i in range(1, len(line_data_list)):
                key_value = line_data_list[i].strip().split(':')
                line_result.append(key_value[1])

            dense_data_list.append(line_result)

    return dense_data_list, label_list


def libsvm_to_dense(libsvm_file_input_path, dense_file_output_path, label_output_path):
    dense_data_list = list()
    label_list = list()
    if os.path.isfile(libsvm_file_input_path):
        handle_one_file(libsvm_file_input_path, dense_data_list, label_list)
    elif os.path.isdir(libsvm_file_input_path):
        libsvm_file_list = os.listdir(libsvm_file_input_path)
        for i in range(0, len(libsvm_file_list)):
            one_file_path = os.path.join(libsvm_file_input_path, libsvm_file_list[i])
            handle_one_file(one_file_path, dense_data_list, label_list)

    with open(dense_file_output_path, 'w') as dense_result:
        for features in dense_data_list:
            dense_result.write(' '.join(features) + '\n')
    with open(label_output_path, 'w') as label_result:
        label_result.write('\n'.join(label_list))


if __name__ == '__main__':
    # libsvm_file_input_path = 'train_data'
    libsvm_file_input_path = 'pre_data'
    dense_file_output_path = 'output_file\\dense.features'
    label_output_path = 'output_file\\dense.labels'
    libsvm_to_dense(libsvm_file_input_path, dense_file_output_path, label_output_path)
