import pandas as pd
import numpy as np
from sklearn import metrics
from pandas import DataFrame
from sklearn.feature_selection import SelectKBest, f_classif

def isNaN(v):
    return v != v


#convert list to column vector
def convert_to_2d(list):
    vector = list.reshape(-1, 1)
    return vector


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


def sklearn_feature_selection_ranks(data_X, data_Y, k):
    feature_selector = SelectKBest(f_classif, k)
    feature_selector.fit(data_X, data_Y)
    feature_names = data_X.columns.values
    mask = feature_selector.get_support()
    new_feature_names = []
    for bool, feature in zip(mask, feature_names):
        if bool:
            new_feature_names.append(feature)
    return new_feature_names


def relief(df: pd.DataFrame, k, times):
    S = df.to_numpy()
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

        for j in range(1, n):
            weights[j] += (minus(chosen[j], closest_different[j]))**2 - (minus(chosen[j], closest_same[j]))**2

    features = df.columns.values
    chosen_features = [(weights[index], features[index]) for index in range(n)]
    chosen_features.sort(key=lambda x: x[0], reverse=True)

    return chosen_features[:k]


def sfs(model, df: DataFrame, train_data: np.ndarray, test_data: np.ndarray, feature_num: int):

    train_labels = convert_to_2d(train_data[:, 0])
    train_data = np.delete(train_data, 0, axis=1)

    y = convert_to_2d(test_data[:, 0])
    test_data = np.delete(test_data, 0, axis=1)

    train = []
    test = []
    added_features = set()

    for j in range(feature_num):
        accuracies = []
        i = 0
        for train_feature, test_feature in zip(train_data.transpose(), test_data.transpose()):

            train_feature = convert_to_2d(train_feature)
            test_feature = convert_to_2d(test_feature)

            if i not in added_features:
                if j == 0:
                    train = convert_to_2d(np.array(train_feature))
                    test = convert_to_2d(np.array(test_feature))
                else:
                    train = np.append(train, train_feature, axis=1)
                    test = np.append(test, test_feature, axis=1)

                model.fit(train, train_labels)
                y_hat = model.predict(test)
                accuracies.append((i, metrics.accuracy_score(y, y_hat)))

                train = np.delete(train, axis=1, obj=train.shape[1]-1)
                test = np.delete(test, axis=1, obj=test.shape[1]-1)
            i += 1

        best_index, _ = max(accuracies, key=lambda x: x[1])

        if j == 0:
            train = convert_to_2d(np.array(train_data[:, best_index]))
            test = convert_to_2d(np.array(test_data[:, best_index]))
        else:
            train_add = convert_to_2d(train_data[:, best_index])
            test_add = convert_to_2d(test_data[:, best_index])
            train = np.append(train, train_add, axis=1)
            test = np.append(test, test_add, axis=1)

        added_features.add(best_index)

    names = df.columns.values
    chosen_features = [(index, names[index]) for index in added_features]
    return set(chosen_features)


def manual_features_remove(data: pd.DataFrame):
    data = data.copy()
    data.drop(columns=['Avg_monthly_expense_on_pets_or_plants'])
    return data
