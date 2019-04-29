import pandas as pd
from FeatureSelection import relief, sfs
from pandas import DataFrame, Series
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns


def get_binary_features(df: DataFrame):
    binary_features = []
    for feature in df:
        if len(get_series_hist(df[feature])) == 2:
            binary_features.append(feature)
    return binary_features


def get_series_hist(series: Series):
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
    df['Age_group'] = df['Age_group'].map({'Below_30': 0, '30-45': 0.5, '45_and_up': 1})
    #binary_names = ['Will_vote_only_large_party', 'Gender', 'Married', 'Looking_at_poles_results', 'Voting_Time', 'Financial_agenda_matters', 'Age_group']


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
    sfs_chosen_features = sfs(clf, data, np_data[0:8000, :], np_data[8000:, :], 30)
    relief_chosen_features = relief(data, np_data, threshold=0.3, times=3)
    chosen_features = sfs_chosen_features.intersection(relief_chosen_features)
    return chosen_features


def standartize(feature_name, data: DataFrame):
    feature = data[feature_name]
    mean = feature.mean()
    std = feature.std()
    data[feature_name] = (data[feature_name] - mean)/std


def normalize(feature_name, data: DataFrame):
    feature = data[feature_name]
    max = feature.max()
    min = feature.min()
    data[feature_name] = (data[feature_name] - min)/(max - min)


def scale(data: DataFrame, features_to_normalize, features_to_standartize):
    for s in features_to_normalize:
        normalize(s, data)
    for s in features_to_standartize:
        standartize(s, data)


def plot_features_hists(data: DataFrame):
    for i, column in enumerate(data):
        name = str(i) + ' ' + column
        data[column].plot(kind='hist')
        plt.title(name)
        plt.show()


def plot_scatters(data: DataFrame):
    for i, column in enumerate(data):
        if i != 0:
            name = str(i) + ' ' + 'vote and ' + column
            plt.title(name)
            plt.scatter(data[column], data['Vote'])
            plt.show()


def count(data: DataFrame, feature_name):
    no = np.zeros(13)
    yes = np.zeros(13)
    for _, sample in data.iterrows():
        n = int(sample['Vote'])
        if sample[feature_name] == 0:
            no[n] += 1
        else:
            yes[n] += 1

    y = [0,1,2,3,4,5,6,7,8,9,10,11,12]
    plt.scatter(no, y)
    plt.scatter(yes, y)
    plt.title(feature_name)
    plt.show()


def plot_vote_to_features_colored(data: DataFrame):
    names = data.columns.values
    for i in range(1, 52):
        sns.pairplot(data.iloc[:, [0, i]], hue='Vote')
        name = 'Vote to ' + str(names[i])
        plt.title(name)
        plt.savefig(name + '.png')
        #plt.show()


def plot_vote_to_features(data: DataFrame):
    names = data.columns.values
    for i in range(1, 52):
        for j in range(0, 12):
            data_labeled = data[data.Vote == j]
            sns.pairplot(data_labeled.iloc[:, [i]])
            name = 'Vote labeled ' + str(j) + ' to ' + str(names[i])
            plt.title(name)
            plt.show()
            plt.savefig(name + '.png')


def arrange_data(df: DataFrame):
    features_to_normalize = [1, 10, 11, 12, 13, 14, 16, 17, 25, 27, 30, 33]
    features_to_standartize = [2, 3, 4, 6, 15, 18, 19, 20, 22, 23, 24, 26, 29, 31, 32]
    data_featues_one_hot = to_numerical_data(df)
    data = data_featues_one_hot.fillna(method='ffill')
    #normalize_names = [data.columns.values[i] for i in features_to_normalize]
    #standartize_names = [data.columns.values[i] for i in features_to_standartize]
    #scale(data, normalize_names, standartize_names)
    return data


if __name__ == '__main__':
    data = pd.read_csv('ElectionsData.csv')
    data = arrange_data(data)
    #data['Vote'].hist()
    #plt.show()
    #plt.savefig('Vote hist.png')
    #plot_vote_to_features(data)
    plot_vote_to_features_colored(data)
