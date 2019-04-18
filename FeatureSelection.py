import pandas as pd
import numpy as np
from sklearn import metrics
from pandas import DataFrame

def isNaN(v):
    return v != v


def minus(val1, val2):
    if isNaN(val1) or isNaN(val2):
        return 0
    if isinstance(val1, str):
        assert isinstance(val2, str)
        if val1 != val2:
            return 1
        else:
            return 0
    else:
        return val1-val2


def euclidean_dist(u, v):
    dist = 0
    for val1, val2 in zip(u, v):
        dist += (minus(val1, val2))**2
    return dist


def relief(df: pd.DataFrame, S: np.ndarray, threshold, times):
    m = S.shape[0]  #num of samples
    n = S.shape[1]  #num of features

    weights = np.zeros(n)
    for _ in range(times):
        instance_index = np.random.randint(0, m)
        chosen = S[instance_index]

        same = [(euclidean_dist(chosen, instance), instance) for i, instance in enumerate(S) if i != instance_index and
                instance[0] == chosen[0]]
        different = [(euclidean_dist(chosen, instance), instance) for instance in S if instance[0] != chosen[0]]

        _, closest_same = min(same, key=lambda x: x[0])
        _, closest_different = min(different, key=lambda x: x[0])

        for j in range(n):
            weights[j] += (minus(chosen[j], closest_different[j]))**2 - (minus(chosen[j], closest_same[j]))**2

    features = df.columns.values
    chosen_features = [(index, features[index]) for index in range(n) if weights[index] > threshold]
    return chosen_features


def sfs(model, df: DataFrame, train_data: np.ndarray, test_data: np.ndarray, feature_num: int):

    train_labels = train_data[:, 0]
    train_labels = train_labels.reshape(len(train_labels), 1)
    train_data = np.delete(train_data, 0, axis=1)

    test_labels = test_data[:, 0]
    test_labels = test_labels.reshape(len(test_labels), 1)
    test_data = np.delete(test_data, 0, axis=1)

    train = []
    test = []
    added_features = set()

    for j in range(feature_num):
        accuracies = []
        i = 0
        for train_feature, test_feature in zip(train_data.transpose(), test_data.transpose()):
            if i in added_features:
                continue

            train = np.append(train, train_feature, axis=1)
            test = np.append(test, test_feature, axis=1)
            if j == 0:
                train = train.reshape(len(train), 1)
                test = test.reshape(len(test), 1)
            else:


            model.fit(train, train_labels)
            y_hat = model.predict(test)
            accuracies.append((i, metrics.accuracy_score(test_labels, y_hat)))
            train = np.delete(train, axis=1, obj=train.shape[1]-1)
            test = np.delete(test, axis=1, obj=test.shape[1]-1)
            i += 1

        best_index, _ = max(accuracies, key=lambda x: x[1])
        train = np.append(train, train_data[:, best_index], axis=1)
        test = np.append(test, test_data[:, best_index], axis=1)

        added_features.add(best_index)


    names = df.columns.values
    chosen_features = [(index, names[index]) for index in added_features]
    return chosen_features
