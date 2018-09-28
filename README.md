#前言

这里是我在使用xgboost算法模型过程中整理出来离线进行模型训练、预测、评估的python源码。

训练和预测的数据来源于公司某天经过过滤并脱敏的数据，其中特征维度是108，样本量在千万左右。


#[python文件]

train_xgboost_model.py —— xgboost模型训练
train_xgboost_config_file.py —— 相比于上面,这里把参数配置抽离出来,通过读取配置文件train_xgboost.conf训练xgboost模型

predict_xgboost.py —— 加载已训练好的模型并对测试数据进行预测，预测结果保存在当前目录的predict.result文件中

evaluation_xgboost.py —— 评估模型.包括AUC值(对应的ROC曲线),F1Score(对应的PR曲线),特征重要度,树模型示例,还有常见的查准率、查全率等

my_util.py —— 常用函数存放,包括加载xgboost模型,加载用于训练或测试的数据并转为DMatrix格式

libsvm2dense.py —— 将libsvm格式的数据转化为dense格式的数据,分别保存到dense.features和dense.labels两个文件中

-------------------------------------------

#[文件夹]
model —— 用于存放训练好的xgboost模型
train_data —— 存放用于训练的数据，数据格式为libsvm
pre_data —— 存放用于预测的数据，数据格式为libsvm

-------------------------------------------

#[文件]
dense.features —— 通过libsvm2dense.py脚本将train_data或pre_data里的libsvm格式数据转化为dense格式，该文件存储特征
dense.label ——  同上，该文件存label列

feature_map.txt —— 使用evaluation_xgboost.py中输出特征重要度图片所需的配置文件
feature_importtance.png —— 使用evaluation_xgboost.py输出的特征重要度图片
tree.png —— 使用evaluation_xgboost.py输出的树示例图
