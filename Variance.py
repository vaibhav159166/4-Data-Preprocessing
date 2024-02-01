"""
Created on Mon Oct  9 08:05:09 2023

@author: Vaibhav Bhorkade

"""

# Zero variance and near zero variance 
# 
import pandas as pd

df=pd.read_csv("C:/datasets/ethnic_diversity.csv")

df.var()
'''
EmpID       3.347435e+16
Zip         2.867558e+08
Salaries    4.441953e+08
age         8.571358e+01
dtype: float64
'''
# here EmpId , zip is normal data 
# Salary is not close to 0
# Similaraly age 

# Check 
df.var()==0
'''
EmpID       False
Zip         False
Salaries    False
age         False
dtype: bool
'''
# None of them equal to 0

df.var(axis=0)==0
'''
EmpID       False
Zip         False
Salaries    False
age         False
dtype: bool'''

#####################################################

import pandas as pd
import numpy as np

df=pd.read_csv("C:/datasets/modified ethnic.csv")

# check for Null values
df.isna().sum()

# Create an imputer create NaN values

from sklearn.impute import SimpleImputer
mean_imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
# check the datafrrame
df['Salaries']=pd.DataFrame(mean_imputer.fit_transform(df[['Salaries']]))

df['Salaries'].isna().sum()