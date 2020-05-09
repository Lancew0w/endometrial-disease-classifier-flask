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

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()   #初始化一个对象sc去对数据集作变换
    sc.fit(X_train)   #用对象去拟合数据集X_train，并且存下来拟合参数
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    
    from sklearn.linear_model import LogisticRegression
    def sigmoid(z):  
        return 1.0 / (1.0 + np.exp(-z)) 
    lr = LogisticRegression(C=1000.0, random_state=0)
    #http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression
    lr.fit(X_train_std, y_train)
    
    lr.predict_proba(X_test_std[0, :].reshape(1, -1))   #计算该预测实例点属于各类的概率
    #Output:array([[  2.05743774e-11,   6.31620264e-02,   9.36837974e-01]])
    
    #验证predict_proba的作用
    c=lr.predict_proba(X_test_std[0, :].reshape(1, -1))
    c[0,0]+c[0,1]+c[0,2]
    #Output:0.99999999999999989
    
    #查看lr模型的特征系数
    lr = LogisticRegression(C=1000.0, random_state=0)
    #http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression
    lr.fit(X_train_std, y_train)
    print('LR 权重：')
    print(lr.coef_)
    #Output:[[-7.34015187 -6.64685581]
    #        [ 2.54373335 -2.3421979 ]
    #        [ 9.46617627  6.44380858]]
    y_pred = lr.predict(X_test)
    num_correct = 0
    acc = 0.1
    for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            num_correct = num_correct + 1
        print(str(i),'Prediction:', y_pred[i], 'Label:', y_test[i])
    acc = num_correct / len(y_pred)
    print('The accuracy of this LR model is:', acc)
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