import pandas as pd
from FeatureSelection import relief, sfs
from pandas import DataFrame
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


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


def chosen_features(data: DataFrame):
    np_data = data.to_numpy()
    clf = DecisionTreeClassifier()
    sfs_chosen_features = sfs(clf, data_featues_one_hot, np_data[0:8000, :], np_data[8000:, :], 30)
    relief_chosen_features = relief(data_featues_one_hot, np_data, threshold=0.3, times=3)
    chosen_features = sfs_chosen_features.intersection(relief_chosen_features)
    return chosen_features


def plot_features_hists(data: DataFrame):
    for i, column in enumerate(data):
        name = str(i) + ' ' + column
        data[column].plot(kind='hist')
        plt.title(name)
        plt.show()


if __name__ == '__main__':
    data = pd.read_csv('ElectionsData.csv')
    data_featues_one_hot = to_numerical_data(data)
    data = data_featues_one_hot.fillna(method='ffill')
    #chosen_features = chosen_features(data)
    plot_features_hists(data)

