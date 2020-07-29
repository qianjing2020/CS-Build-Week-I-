# with open('Build-Week1/data/data_banknote_authentication.txt') as f:
#     line = f.readlines()
#     for iline in line:
#         print(iline)
import pandas as pd
import numpy as np

def get_test_data():
    file = 'Build-Week1/data/data_banknote_authentication.txt'
    f = open(file,'r')
    lst = []
    data = np.loadtxt(f, dtype = {'name':('variance', 'skewness', 'curtosis', 'entropy','class'), 'formats': ('f4', 'f4', 'f4', 'f4', 'f4')})
    test_data_0 = data[0:20, :]
    # get some in class 1
    test_data_1 = data[-20:, :]
    # combine 
    small_data = np.concatenate((test_data_0, test_data_1), axis = 0)
    return small_data

small_data = get_test_data()
print(small_data)