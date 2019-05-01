import pandas as pd
import numpy as np
import math
from imputations import DistirbutionImputator

B_list = list(np.random.normal(10, 4, 4))
B_list.append(np.nan)
B_list.append(np.nan)
B_list.append(np.nan)
B_list.append(np.nan)
B_list.extend(np.random.normal(8, 8, 4))
B_list.append(np.nan)
B_list.append(np.nan)
B_list.append(np.nan)
B_list.append(np.nan)
B_list.extend(np.random.normal(100, 8, 4))
B_list.append(np.nan)
B_list.append(np.nan)
B_list.append(np.nan)
B_list.append(np.nan)

C_list = list(np.random.normal(-10, 4, 4))
C_list.append(np.nan)
C_list.append(np.nan)
C_list.append(np.nan)
C_list.append(np.nan)
C_list.extend(np.random.normal(-88, 8, 4))
C_list.append(np.nan)
C_list.append(np.nan)
C_list.append(np.nan)
C_list.append(np.nan)
C_list.extend(np.random.normal(-200, 8, 4))
C_list.append(np.nan)
C_list.append(np.nan)
C_list.append(np.nan)
C_list.append(np.nan)


data = pd.DataFrame(data=({'L': ['a','a','a','a','a','a','a','a','b','b','b','b','b','b','b','b', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c'],
                           'B': B_list,
                           'C': C_list}))


data = data.sort_values(by='L')
print(data)
imp = DistirbutionImputator()
imp.fit(data)
imp.fill_nans(data)
print(data)
