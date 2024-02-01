# Normalization
import pandas as pd
import numpy as np
ethnic=pd.read_csv("C:/datasets/ethnic_diversity.csv")
# columns
ethnic.columns
# There are some columns which is not useful then drop it
ethnic.drop(columns={'Employee_Name','EmpID','Zip'},axis=1,inplace=True)
# Now read minimum and maximum values of salaries and age
a1=ethnic.describe()
# Check a1 data frame in variable explorer,
# You find minimum salary is 0 and max is 108304
# Same way check for age, there is huge difference
# in min and max value. Hence we are going for normalization
# First we will have to convert non-numeric data to label encoding
ethnic=pd.get_dummies(ethnic,drop_first=True)
# Normalization function written where ethnic arguments is passed
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
df_norm=norm_func(ethnic)
b=df_norm.describe()
# If you will observe the b frame,
# It has dimensions 8,81
# Earlier in a they were 8,11, it is because all non-numeric
# Data has been converted to numeric using label encoding
