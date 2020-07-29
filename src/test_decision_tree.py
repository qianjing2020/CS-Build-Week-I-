# with open('Build-Week1/data/data_banknote_authentication.txt') as f:
#     line = f.readlines()
#     for iline in line:
#         print(iline)
import pandas as pd
import numpy as np
import pandas as pd
def get_iris_data():
    file = 'Build-Week1/data/iris.data'
    df = pd.read_csv(file, header = None, skiprows=0)
    header = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
    data = df.to_numpy()
    return df, header


data, header = get_iris_data()

