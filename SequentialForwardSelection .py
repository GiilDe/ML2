from pandas import DataFrame
import pandas as pd
import numpy as np
from sklearn import metrics


def sfs(model, df: DataFrame, train_data: np.ndarray, test_data: np.ndarray, feature_num: int):

    train_labels = train_data[:, 0]
    train_data = np.delete(train_data, 0, axis=1)

    test_labels = test_data[:, 0]
    test_data = np.delete(test_data, 0, axis=1)

    features = np.array()
    added_features = set()

    for _ in range(feature_num):
        accuracies = []
        for i, feature in enumerate(train_data.transpose()):
            if i in added_features:
                continue

            features = np.append(features, feature)
            model.fit(features, train_labels)
            y_hat = model.predict(test_data)
            accuracies.append((i, metrics.accuracy_score(test_labels, y_hat)))
            features = np.delete(features, axis=1, features.shape()[1]-1)

        best_index, _ = max(accuracies, key=lambda x: x[1])
        features = np.append(features, train_data[:, best_index])
        added_features.add(best_index)

    names = df.columns.values
    chosen_features = [(index, names[index]) for index in added_features]
    return chosen_features