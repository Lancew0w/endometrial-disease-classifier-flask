import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle
import os
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report


def train(filename):

    df = pd.read_csv(filename)
    X = df.drop('Outcome', axis=1).values
    y = df['Outcome'].values
    test_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # 将参数写成字典下形式
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',  # 设置提升类型
        'objective': 'regression', # 目标函数
        'metric': {'l2', 'auc'},  # 评估函数
        'num_leaves': 50,   # 叶子节点数
        'learning_rate': 0.005,  # 学习速率
        'feature_fraction': 0.9, # 建树的特征选择比例
        'bagging_fraction': 0.8, # 建树的样本采样比例
        'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
        'verbose': 1 # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    }
   
    print('Start training...')
    gbm = lgb.train(params,lgb_train,num_boost_round=100,valid_sets=lgb_eval,early_stopping_rounds=5)

    print('Save model...')
    gbm.save_model('model.txt')

    print('Start predicting...')
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    #print('The f1 of prediction is:', f1_score(y_test, y_pred , average='weighted'))  
    for i in range(len(y_pred)):
        print(str(i),'Prediction:', y_pred[i], 'Label:', y_test[i])
    #print(y_pred)
    #print(y_test)
    print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
    #print(classification_report(y_test,y_pred))

    return 0


def test(filename):
    with open(os.path.join('static', 'model.pkl'), 'rb') as file:
        knn = pickle.load(file)
    df = pd.read_csv(filename)
    InputData = df.drop('Outcome', axis=1).values
    OutputData = knn.predict(InputData)
    new = np.append(InputData, OutputData.reshape(-1, 1), axis=1)

    pd.DataFrame(new).to_csv(os.path.join('static', 'result.csv'), header=None, index=None)
    return OutputData


def test_one(argslist):
    with open(os.path.join('static', 'model.pkl'), 'rb') as file:
        knn = pickle.load(file)
    InputData = np.array(argslist).reshape(1, -1)
    OutputData = knn.predict(InputData)
    print('The type is', int(OutputData))
    return int(OutputData)


def self_test(filename):
    with open(os.path.join('static', 'model.pkl'), 'rb') as file:
        knn = pickle.load(file)
    df = pd.read_csv(filename)
    InputData = df.drop('Outcome', axis=1).values
    OutputData = knn.predict(InputData)
    y = df['Outcome'].values
    correct = 0
    for i in range(len(OutputData)):
        if y[i] == OutputData[i]:
            correct = correct + 1
        else:
            print('right:', y[i], 'wrong:', OutputData[i])
    correct_ratio = 0.0
    correct_ratio = correct / len(OutputData)
    print(correct_ratio)


if __name__ == '__main__':
    pass
    train('train.csv')
    #test('test.csv')
    #test_one([1,14,7.8,1,1,2,2,2,3])
    #self_test('selftest.csv')