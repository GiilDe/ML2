import pandas as pd
from data_handling import to_numerical_data, split_data, scale_all, X_Y_2_XY, XY_2_X_Y
from imputations import impute_train_X, impute_test_and_validation
from outliers_detection import clean_and_correct_train_X
from TestPreformance import test_data_quality
from FeatureSelection import manual_features_remove, sklearn_feature_selection_ranks, sfs, relief


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
    print('select_K_best_features: ', select_K_best_features)
    select_K_best_train_X = train_X[select_K_best_features]
    select_K_best_test_X = test_X[select_K_best_features]
    select_K_best_train_XY = X_Y_2_XY(train_X, train_Y)
    select_K_best_train_XY.to_csv('select_K_best_train_XY')
    print(test_data_quality(select_K_best_train_X, train_Y, select_K_best_test_X, test_Y))
    relief_features = relief(train_XY, 36, 100)