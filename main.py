import pandas as pd
from pandas.plotting import scatter_matrix
from Relief import relief
from sklearn.mixture import GaussianMixture


if __name__ == '__main__':
    data = pd.read_csv('ElectionsData.csv')
    #pd.DataFrame.replace(data, 'Yes', 1, inplace=True)
    #pd.DataFrame.replace(data, 'No', 0, inplace=True)

    print(data.shape)
    data_one_hot = pd.get_dummies(data)
    #data_ndarray = data.to_numpy()
    #chosen_features = relief(data_ndarray, 1000000000, 3)
    #scatter_matrix(data)
    gm = GaussianMixture()
    gm.fit(data_one_hot)
    data_one_hot_filled = gm.predict(data_one_hot)