import pandas as pd
import numpy as np
# from main import to_numerical_data
from sklearn import preprocessing
import math


class DistirbutionImputator:
    def __init__(self):
        self.dict = dict()

    def fit(self, data: pd.DataFrame):  # asserts first column is the label column, also sorts data by label
        label_name = data.columns[0]
        data = data.sort_values(by=[str(label_name)])
        dividing_by_label = []
        current_label = data.iloc[0, 0]
        for i in range(len(data)):
            if data.iloc[i, 0] != current_label:
                dividing_by_label.append(i)
                current_label = data.iloc[i, 0]

        dividing_by_label.append(len(data))
        # print('dividing_by_label: ', dividing_by_label)

        i = 0
        for column_index in range(1, len(data.columns)):
            for j in dividing_by_label:
                mean = data.iloc[i:j, column_index].mean()
                std = data.iloc[i:j, column_index].std()
                self.dict.update({(data.iloc[j-1, 0], column_index): (mean, std)})
                i = j
        print('self.dict:', self.dict)


    def fill_nans(self, data, data_is_with_label_column = True):
        if data_is_with_label_column:
            add_to_column = 0
        else:
            add_to_column = 1
        nans_indeces = np.where(np.asanyarray(pd.isnull(data)))
        print('There are ', len(nans_indeces[0]), ' nans')
        mean_nan_counter = 0
        std_nan_counter = 0
        not_found_counter = 0
        for row, column in zip(nans_indeces[0], nans_indeces[1]):
            mean_std_tuple = self.dict.get((data.iloc[row, 0], column + add_to_column))
            # print('(', data.iloc[row, 0], ',', column + 1, ')', mean_std_tuple, ' for ', row, ', ', column)
            assert math.isnan(data.iloc[row, column])
            if mean_std_tuple is None:
                not_found_counter += 1
                mean = 0
                std = 1
            else:
                mean = mean_std_tuple[0]
                std = mean_std_tuple[1]
            if math.isnan(mean):
                mean_nan_counter += 1
                mean = 0
            if math.isnan(std):
                std_nan_counter += 1
                std = 1

            data.iloc[row, column] = np.random.normal(mean, std)
            assert not(math.isnan(data.iloc[row, column]))
            # print('_:_after_:_', data.iloc[row, column])
        nans_indeces = np.where(np.asanyarray(pd.isnull(data)))
        print('There are ', len(nans_indeces[0]), ' nans')
        print('not_found_counter: ', not_found_counter, 'mean_nan_counter: ', mean_nan_counter, 'std_nan_counter: ', std_nan_counter)
        return data



# all_data = pd.read_csv('ElectionsData.csv')
# all_data = to_numerical_data(all_data)
# all_data = all_data
# di = DistirbutionImputator()
# di.fit(all_data)
#
# di.fill_nans(all_data)
