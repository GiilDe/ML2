import pandas as pd
from pandas.plotting import scatter_matrix
# from FeatureSelection import relief, sfs
from sklearn.mixture import GaussianMixture
from scipy.linalg import svd
from pandas import DataFrame
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from fancyimpute_.fancyimpute.knn import KNN

def count_bad_samples(df: DataFrame):
    bad = 0
    for _, sample in df.iterrows():
        if sample.count() != 38:
            bad += 1
    return bad


if __name__ == '__main__':
    data = pd.read_csv('ElectionsData.csv')
    X_filled_knn = KNN(k=3).fit_transform(data[1:])
    print(X_filled_knn)
    data_features_one_hot = pd.get_dummies(data, )














    #
    # #pd.DataFrame.replace(data, 'Yes', 1, inplace=True)
    # #pd.DataFrame.replace(data, 'No', 0, inplace=True)
    # shape = data.shape
    # #bad = count_bad_samples(data)
    # data_filled = data.fillna(method='ffill')
    # data_one_hot = pd.get_dummies(data_filled)
    #
    # data = pd.read_csv('ElectionsData.csv')
    # data_array = data_one_hot.to_numpy()
    # #u, s, vh = svd(data_array)
    # #chosen_features = relief(data_one_hot, data_array, 8, 3)
    #
    # #gm = GaussianMixture()
    # #gm.fit(data_one_hot)
    # #data_one_hot_filled = gm.predict(data_one_hot)
    # data_featues_one_hot = data_filled.drop(columns='Vote')
    # data_featues_one_hot = pd.get_dummies(data_featues_one_hot)
    # data_featues_one_hot.insert(0, 'Vote', data_filled['Vote'])
    # data_featues_one_hot['Vote'] = data_featues_one_hot['Vote'].map({
    #     'Khakis': 0, 'Oranges': 1, 'Purples': 2, 'Turquoises': 3, 'Yellows': 4, 'Blues': 5, 'Whites': 6,
    #     'Greens': 7, 'Violets': 8, 'Browns': 9, 'Greys': 10, 'Pinks': 11, 'Reds': 12,
    # })
    #
    # normelized_feature = data_featues_one_hot[1:]
    #
    # X_filled_knn = KNN(k=3).fit_transform(data_featues_one_hot)
    # print(X_filled_knn)
    #
    # np_data = data_featues_one_hot.to_numpy()
    # clf = DecisionTreeClassifier()
    # # chosen_features = sfs(clf, data_featues_one_hot, np_data[0:8000, :], np_data[8000:, :], 20)
    #
