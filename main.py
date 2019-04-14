import pandas as pd
from pandas.plotting import scatter_matrix
from Relief import relief
from sklearn.mixture import GaussianMixture
from scipy.linalg import svd


def count_bad_samples(df: pd.DataFrame):
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
    #chosen_features = relief(data_array, 1000000000, 3)
    #scatter_matrix(data_one_hot)
    #gm = GaussianMixture()
    #gm.fit(data_one_hot)
    #data_one_hot_filled = gm.predict(data_one_hot)
