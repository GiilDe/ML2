import pandas as pd
from data_initial_hanedling import to_numerical_data, split_data, scale_all
from imputations import impute_train_X, impute_test_and_validation
from outliers_detection import clean_and_correct_train_X
from TestPreformance import test_data_quality
import pickle
if __name__ == '__main__':
    data = pd.read_csv('ElectionsData.csv')
    data = data.iloc[0:300, :]
    data_featues_one_hot = to_numerical_data(data)
    data = data_featues_one_hot.fillna(method='ffill')
    train_X, train_Y, validation_X, validation_Y, test_X, test_Y = split_data(data)
    train_X, train_Y = clean_and_correct_train_X(train_X, train_Y)
    train_X = impute_train_X(train_X, train_Y)
    validation_X, test_X = impute_test_and_validation(train_X, validation_X, test_X)
    train_XY = train_X.copy()
    train_XY.insert(loc=0, column='Vote', value=train_Y)
    validation_XY = validation_X.copy()
    validation_XY.insert(loc=0, column='Vote', value=validation_Y)
    test_XY = test_X.copy()
    test_XY.insert(loc=0, column='Vote', value=test_Y)
    train_XY.to_csv()
    validation_XY.to_csv()
    test_XY.to_csv()
    train_XY, validation_XY, test_XY = scale_all(train_XY, validation_XY, test_XY)
    # now the data is ready for feature selection
    print(test_data_quality(train_X, train_Y, test_X, test_Y))
