import pandas as pd
from pandas.plotting import scatter_matrix
from FeatureSelection import relief, sfs
from sklearn.mixture import GaussianMixture
from scipy.linalg import svd
from pandas import DataFrame
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def count_bad_samples(df: DataFrame):
    bad = 0
    for _, sample in df.iterrows():
        if sample.count() != 38:
            bad += 1
    return bad


if __name__ == '__main__':
    data = pd.read_csv('ElectionsData.csv')
    #pd.DataFrame.replace(data, 'Yes', 1, inplace=True)
    #pd.DataFrame.replace(data, 'No', 0, inplace=True)
    shape = data.shape
    #bad = count_bad_samples(data)
    data_filled = data.fillna(method='ffill')
    data_one_hot = pd.get_dummies(data_filled)

    data_array = data_one_hot.to_numpy()
    #u, s, vh = svd(data_array)
    #chosen_features = relief(data_one_hot, data_array, 8, 3)

    #gm = GaussianMixture()
    #gm.fit(data_one_hot)
    #data_one_hot_filled = gm.predict(data_one_hot)
    data_featues_one_hot = data_filled.drop(columns='Vote')
    data_featues_one_hot = pd.get_dummies(data_featues_one_hot)
    data_featues_one_hot.insert(0, 'Vote', data_filled['Vote'])
    data_featues_one_hot['Vote'] = data_featues_one_hot['Vote'].map({
        'Khakis': 0, 'Oranges': 1, 'Purples': 2, 'Turquoises': 3, 'Yellows': 4, 'Blues': 5, 'Whites': 6,
        'Greens': 7, 'Violets': 8, 'Browns': 9, 'Greys': 10, 'Pinks': 11, 'Reds': 12,
    })

    np_data = data_featues_one_hot.to_numpy()
    clf = DecisionTreeClassifier()
    chosen_features = sfs(clf, data_featues_one_hot, np_data[0:8000, :], np_data[8000:, :], 20)

