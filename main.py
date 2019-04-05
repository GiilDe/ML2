import pandas as pd
import Relief

if __name__ == '__main__':
    data = pd.read_csv('ElectionsData.csv')
    data_ndarray = data.to_numpy()
    chosen_features = Relief.relief(data_ndarray, 1000000000, 3)
    X = 5