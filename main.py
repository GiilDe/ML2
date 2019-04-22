import pandas as pd
from pandas.plotting import scatter_matrix
from FeatureSelection import relief, sfs
from sklearn.mixture import GaussianMixture
from scipy.linalg import svd
from pandas import DataFrame
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from fancyimpute_.fancyimpute.knn import KNN
from TestPreformance import test_data_quality
from sklearn.impute import SimpleImputer
from imputations import DistirbutionImputator

def remove_bad_samples(df: DataFrame):
    categorials = ['Main_transportation', 'Occupation', 'Most_Important_Issue']
    for i, sample in df.iterrows():
        cat_values = set(sample[feature] for feature in categorials)
        if np.nan in cat_values:
            df.drop(axis=0, index=i)


def to_numerical_data(data: DataFrame):
    remove_bad_samples(data)
    convert_binary(data)
    data_featues_one_hot = data.drop(columns='Vote')
    data_featues_one_hot = pd.get_dummies(data_featues_one_hot)
    data_featues_one_hot.insert(0, 'Vote', data['Vote'])
    data_featues_one_hot['Vote'] = data_featues_one_hot['Vote'].map({
        'Khakis': 0, 'Oranges': 1, 'Purples': 2, 'Turquoises': 3, 'Yellows': 4, 'Blues': 5, 'Whites': 6,
        'Greens': 7, 'Violets': 8, 'Browns': 9, 'Greys': 10, 'Pinks': 11, 'Reds': 12,
    })
    return data_featues_one_hot


def get_binary_features(df: DataFrame):
    binary_features = []
    for feature in df:
        n = get_series_hist(df[feature])
        if len(n) == 2:
            binary_features.append(feature)
    return binary_features


def get_series_hist(series: pd.Series):
    values = set()
    for value in series:
        if value is not np.nan:
            values.add(value)
    return values


def convert_binary(df: DataFrame):
    #pd.DataFrame.replace(data, 'Yes', 1, inplace=True)
    #pd.DataFrame.replace(data, 'No', 0, inplace=True)
    df['Will_vote_only_large_party'] = df['Will_vote_only_large_party'].map({'No': 0, 'Yes': 1, 'Maybe': 0.5})
    df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
    df['Married'] = df['Married'].map({'No': 0, 'Yes': 1})
    df['Looking_at_poles_results'] = df['Looking_at_poles_results'].map({'No': 0, 'Yes': 1})
    df['Voting_Time'] = df['Voting_Time'].map({'By_16:00': 0, 'After_16:00': 1})
    df['Financial_agenda_matters'] = df['Financial_agenda_matters'].map({'No': 0, 'Yes': 1})
    df['Age_group'] = df['Age_group'].map({'Below_30': 0, '30-45': 0.5, '45_and__up': 1})


#bad sample is a sample with nan in a categorial values
def count_bad_samples(df: DataFrame):
    categorials = ['Main_transportation', 'Occupation', 'Most_Important_Issue']
    bad = 0
    for _, sample in df.iterrows():
        cat_values = set(sample[feature] for feature in categorials)
        if np.nan in cat_values:
            bad += 1
    return bad


def split_data(all_data):
    num_of_examples = all_data.shape[0]
    num_of_examples_in_train = int(num_of_examples * 0.8)
    # TODO: split data wisely
    all_data = pd.DataFrame(all_data)
    train_data = all_data.iloc[0:num_of_examples_in_train]
    test_data = all_data.iloc[num_of_examples_in_train + 1:]
    train_data_X = train_data.iloc[:, 1:]
    train_data_Y = train_data.iloc[:, 0]
    test_data_X = test_data.iloc[:, 1:]
    test_data_Y = test_data.iloc[:, 0]
    return train_data_X, train_data_Y, test_data_X, test_data_Y


if __name__ == '__main__':
    all_data = pd.read_csv('ElectionsData.csv')
    all_data = to_numerical_data(all_data)

    #___________________KNN_imputation___________________# # illegal
    #
    # KNNed_data = KNN(k=3).fit_transform(all_data)
    # train_data_X, train_data_Y, test_data_X, test_data_Y = split_data(all_data)
    # scaler = preprocessing.StandardScaler().fit(train_data_X)
    # scaled_train_data_X = pd.DataFrame(scaler.transform(train_data_X))
    # scaled_test_data_X = pd.DataFrame(scaler.transform(test_data_X))
    # print(test_data_quality(scaled_train_data_X, train_data_Y, scaled_test_data_X, test_data_Y))
    #
    # ___________________Simple_Imputer___________________#

    # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    # train_data_X, train_data_Y, test_data_X, test_data_Y = split_data(all_data)
    # scaler = preprocessing.StandardScaler().fit(train_data_X)
    # scaled_train_data_X = pd.DataFrame(scaler.transform(train_data_X))
    # imp.fit(scaled_train_data_X)
    # imputed_scaled_train_data_X = imp.transform(scaled_train_data_X)
    # imputed_scaled_test_data_X = imp.transform(pd.DataFrame(scaler.transform(test_data_X)))
    # print(test_data_quality(imputed_scaled_train_data_X, train_data_Y, imputed_scaled_test_data_X, test_data_Y))

    # ___________________Distribution_Imputer___________________#
    train_data_X, train_data_Y, test_data_X, test_data_Y = split_data(all_data)

    train_data = train_data_X.copy()
    train_data.insert(loc=0, column='Vote', value=train_data_Y)
    imp = DistirbutionImputator()
    imp.fit(train_data)
    imputed_train_data_X = imp.fill_nans(train_data_X, data_is_with_label_column=False)
    scaler = preprocessing.StandardScaler().fit(imputed_train_data_X)
    imputed_scaled_train_data_X = pd.DataFrame(scaler.transform(imputed_train_data_X))
    simple_imputer = SimpleImputer()
    simple_imputer.fit(train_data_X)
    imputed_scaled_test_data_X = scaler.transform(simple_imputer.transform(pd.DataFrame(test_data_X)))
    print(test_data_quality(imputed_scaled_train_data_X, train_data_Y, imputed_scaled_test_data_X, test_data_Y))

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
