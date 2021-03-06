import pandas as pd
from data_handling import to_numerical_data, split_data, scale_all, X_Y_2_XY, XY_2_X_Y
from imputations import impute_train_X, impute_test_and_validation
from outliers_detection import clean_and_correct_train_X
from TestPreformance import test_data_quality
from FeatureSelection import manual_features_remove, sklearn_feature_selection_ranks, sfs, relief
import pickle

load = True

if __name__ == '__main__':

    # validation_XY = pd.read_csv('validation_XY')
    # validation_X = pd.read_csv('validation_X')
    # validation_Y = pd.read_csv('validation_Y')
    #
    # validation_XY.insert(loc=0, column='0', value=validation_Y, allow_duplicates=True)
    #

    if load:
        train_XY = pd.read_csv('train_XY')
        validation_XY = pd.read_csv('validation_XY')
        test_XY = pd.read_csv('test_XY')
    else:
        data = pd.read_csv('ElectionsData.csv')
        # data = data.iloc[:300, :]
        data = to_numerical_data(data)
        # data = data_featues_one_hot.fillna(method='ffill')
        train_X, train_Y, validation_X, validation_Y, test_X, test_Y = split_data(data)
        train_XY = X_Y_2_XY(train_X, train_Y)
        validation_XY = X_Y_2_XY(validation_X, validation_Y)
        test_XY = X_Y_2_XY(test_X, test_Y)
        train_XY = impute_train_X(train_XY)
        train_XY = clean_and_correct_train_X(train_XY)
        train_XY, validation_XY, test_XY = scale_all(train_XY, validation_XY, test_XY)
        validation_XY, test_XY = impute_test_and_validation(train_XY, validation_XY, test_XY)
        train_XY.to_csv('train_XY')
        validation_XY.to_csv('validation_XY')
        test_XY.to_csv('test_XY')

    train_X = train_XY.iloc[:, 1:]
    train_Y = train_XY.iloc[:, 0]
    validation_X = validation_XY.iloc[:, 1:]
    validation_Y = validation_XY.iloc[:, 0]
    test_X = test_XY.iloc[:, 1:]
    test_Y = test_XY.iloc[:, 0]
    """feature selection"""
    select_K_best_features = sklearn_feature_selection_ranks(train_X, train_Y, 36)
    pickle.dump(select_K_best_features, open('select_K_best_features', 'wb'))
    print('select_36_best_features: ', select_K_best_features)
    select_K_best_train_X = train_X[select_K_best_features]
    select_K_best_validation_X = validation_X[select_K_best_features]
    select_K_best_validation_XY = X_Y_2_XY(select_K_best_train_X, train_Y)
    print(test_data_quality(select_K_best_train_X, train_Y, select_K_best_validation_X, validation_Y))
    relief_features = relief(train_XY, 36, 100)
    pickle.dump(relief_features, open('relief_36_features', 'wb'))
    relief_features_train_X = train_X[relief_features]
    relief_features_validation_X = validation_X[relief_features]
    print(test_data_quality(relief_features_train_X, train_Y, relief_features_validation_X, validation_Y))
    from sklearn.neighbors import KNeighborsClassifier
    sfs_features = sfs(KNeighborsClassifier(n=3), train_XY, train_XY.to_numpy(), validation_XY.to_numpy(), 36)
    pickle.dump(sfs_features, open('sfs_36_features', 'wb'))
    sfs_features_train_X = train_X[sfs_features]
    sfs_features_validation_X = validation_X[sfs_features]
    print(test_data_quality(sfs_features_train_X, train_Y, sfs_features_validation_X, validation_Y))
