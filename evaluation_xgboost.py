# by zijingrong
# -*- coding:utf-8 -*-
import my_util
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn import metrics


def calculate_tp_tn_fp_fn(labels, pre_labels):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for i in range(len(labels)):
        if int(labels[i]) == 1 and int(pre_labels[i]) == 1:
            true_positive = true_positive + 1
        if int(labels[i]) == 0 and int(pre_labels[i]) == 0:
            true_negative = true_negative + 1
        if int(labels[i]) == 1 and int(pre_labels[i]) == 0:
            false_positive = false_positive + 1
        if int(labels[i]) == 0 and int(pre_labels[i]) == 1:
            false_negative = false_negative + 1
        labels[i] = int(labels[i])
        pre_labels[i] = int(pre_labels[i])
    return true_positive, true_negative, false_positive, false_negative


def plot_roc_curve(labels, pre_scores):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, the_threshold = metrics.roc_curve(labels, pre_scores)  # 计算真正率和假正率
    roc_auc = metrics.auc(fpr, tpr)  # 计算auc的值

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def plot_precision_recall_curve(labels, pre_scores):
    average_precision = metrics.average_precision_score(labels, pre_scores)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    precision, recall, _ = metrics.precision_recall_curve(labels, pre_scores)

    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.show()


def model_evaluation(model_input, test_data_input, threshold_input):
    # prediction
    pre_scores = model_input.predict(test_data_input)
    print ('There are ' + str(len(pre_scores)) + ' rows of test_data.')

    # get the label of data
    label_list = test_data_input.get_label()

    # Print model report:
    print ("\n-------------------------------------")
    print ("Model Report")
    pre_label = []
    for p in pre_scores:
        if p > threshold_input:
            pre_label.append(1)
        else:
            pre_label.append(0)

    print ("The count of labels: %d" % len(label_list))
    # TP, TN, FP, FN
    true_positive, true_negative, false_positive, false_negative = calculate_tp_tn_fp_fn(label_list, pre_label)
    print(true_positive, true_negative, false_positive, false_negative)
    print ("\n-------------------")
    # 准确率
    print('(TP+TN)/(TP+TN+FP+FN) = accuracy : ',
          (true_positive + true_negative) / float(true_positive + true_negative + false_positive + false_negative))
    print('metrics.accuracy_score : ', metrics.accuracy_score(label_list, pre_label))
    print ("\n-------------------")
    # 查准率
    precision = true_positive / float(true_positive + false_positive)
    print('TP/(TP+FP) = precision : ', precision)
    print('metrics.precision_score : ', metrics.precision_score(label_list, pre_label))
    print ("\n-------------------")
    # 查全率
    recall = true_positive / float(true_positive + false_negative)
    print('TP/(TP+FN) = recall : ', recall)
    print('metrics.recall_score : ', metrics.recall_score(label_list, pre_label))
    print ("\n-------------------")
    # B = 1
    # f_score = ((1 + B ** 2) * precision * recall) / ((B ** 2 * precision) + recall)
    # print('f1_score:', f_score)  # 准确率与召回率的调和平均指标
    print('metrics.f1_score:', metrics.f1_score(label_list, pre_label))
    print ("\n-------------------")
    # B = 0.5
    # f_score = ((1 + B ** 2) * precision * recall) / ((B ** 2 * precision) + recall)
    # print('beta=0.5:', f_score)  # 准确率与召回率的调和平均指标
    # print('metrics.fbeta_score:', metrics.fbeta_score(label_list, pre_label, beta=0.5))

    # calculate AUC score
    print ("AUC Score : %f" % metrics.roc_auc_score(label_list, pre_scores))

    # plot the ROC curve
    plot_roc_curve(label_list, pre_scores)

    # plot the Precision-Recall curve
    plot_precision_recall_curve(label_list, pre_scores)

    # plot features' importance
    fig_importance, ax = plt.subplots(figsize=(15, 15))
    xgb.plot_importance(test_model, height=0.5, ax=ax, max_num_features=64)
    fig_importance.savefig('feature_importance.png')
    plt.show()

    # plot tree
    fig_tree, ax_tree = plt.subplots(figsize=(20, 20))
    xgb.plot_tree(test_model, ax=ax_tree, fmap=feature_map_p)
    fig_tree.savefig('tree.png')
    plt.show()


if __name__ == '__main__':
    threshold = 0.5
    test_model_p = 'model\\LocalFile'
    test_data_p = 'pre_data\\part-00000'
    feature_map_p = 'feature_map.txt'

    test_model = my_util.load_model(test_model_p)
    test_data = my_util.load_libsvm_data(test_data_p)
    model_evaluation(test_model, test_data, threshold)
