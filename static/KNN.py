import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle
import os


def train(filename):

    df = pd.read_csv(filename)
    X = df.drop('Outcome', axis=1).values
    y = df['Outcome'].values
    test_size = 0.4
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    neighbors = np.arange(1, 20)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy  = np.empty(len(neighbors))
    print('Start Training')
    for i, k in enumerate(neighbors):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        train_accuracy[i] = knn.score(X_train, y_train)
        test_accuracy[i] = knn.score(X_test, y_test)
        # tmp_y_predict = knn.predict(X_test)
        # error_index = np.nonzero(y_test - tmp_y_predict)[0]
    best = list(test_accuracy).index(max(list(test_accuracy))) + 1
    knn = KNeighborsClassifier(n_neighbors=best)
    knn.fit(X_train, y_train)
    model_name = 'model.pkl'
    print('Completed')

    #with open(os.path.join('static', model_name), 'wb') as file:
    #    pickle.dump(knn, file)
    # print('The best train accuracy is', '%.2f' % float(max(list(train_accuracy)))*100)
    best_acc = 100 * float(max(list(test_accuracy)))
    print('The best test accuracy is', '%.2f' % best_acc, '%')

    plt.title('KNN Accuracy')
    plt.plot(neighbors, train_accuracy, label='Train Accuracy')
    plt.plot(neighbors, test_accuracy, label='Test Accuracy')
    plt.legend()
    plt.xlabel('Number of neighbors')
    plt.ylabel('Accuracy')
    plt.show()
    plt.savefig('1.png')

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