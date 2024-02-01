# -*- coding: utf-8 -*-

"""
Created on Fri Oct  6 08:19:26 2023

@author: Vaibhav Bhorkade

EDA
Data pre-processing

"""

"""
Business Objective
Minimize : prediction error
Maximaze : appropriate predicting model or technique

Business constraints : Develop appropriate predicting model 
"""
import pandas as pd

df=pd.read_csv("C:/datasets/Boston.csv")

''' EDA '''
# Continous data
# unstructred data
df.head()

# Shape
df.shape
# 506,15

df.dtypes
'''
Unnamed: 0      int64
crim            int32
zn            float64
indus         float64
chas            int64
nox           float64
rm            float64
age           float64
dis           float64
rad             int64
tax             int64
ptratio       float64
black         float64
lstat         float64
medv          float64
dtype: object
'''

df.crim=df.crim.astype(int)

df.columns
'''Index(['Unnamed: 0', 'crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis',
       'rad', 'tax', 'ptratio', 'black', 'lstat', 'medv'],
      dtype='object')
'''

# Five number summary
df.describe()

df.isnull()
# False

df.isnull().sum()
# 0

df.dropna()

df.isnull().sum()

# mean
df.mean()

#####################################################

import seaborn as sns
import matplotlib.pyplot as plt

# histplot

sns.histplot(df['crim'],kde=True)
# data is right-skew and the not normallly distributed

sns.histplot(df['black'],kde=True)
# data is left-skew and the not normallly distributed

sns.histplot(df,kde=True)
#The data is showing the skewness 

######################################################

# box plot on column
sns.boxplot(df.crim)
# There is outliers

# box plot on column
sns.boxplot(df.zn)

# box plot on column
sns.boxplot(df.indus)

# box plot on column
sns.boxplot(df.chas)

sns.boxplot(df.rm)

sns.boxplot(df.age)

# box plot on all dataframe
sns.boxplot(data=df)
# There is outliers on crim,zn,indus,dis,black columns

# Scatterplot on column
sns.scatterplot(df.crim)

# Scatterplot on column
sns.scatterplot(df.zn)

# Scatterplot on column
sns.scatterplot(df.indus)

# Scatterplot on dataframe
sns.scatterplot(data=df)

#############################################################
df.mean()

# median
df.median()

# Standard deviation
df.std()
''' Standard deviation of the tax and black is more '''

##########################################

# Identify the duplicates
 
duplicate=df.duplicated()
# Output of this function is single columns
# if there is duplicate records output- True
# if there is no duplicate records output-False
# Series will be created

duplicate
sum(duplicate)
# output will be zero


df_new1=df.drop_duplicates()
duplicate2=df.duplicated()
duplicate2

sum(duplicate2)

# Outliers treatment
import pandas as pd
import seaborn as sns

sns.boxplot(df.black)
# There are outliers

sns.boxplot(df.tax)
# There is no outliers

IQR=df.black.quantile(0.75)-df.black.quantile(0.25)
# Have observed IQR in variable explorer
# no,because IQR is in capital letters
# treated as constant
                       
IQR
# but if we try as I,Iqr, or iqr then it is showing

lower_limit=df.black.quantile(0.25)-1.5*IQR

upper_limit=df.black.quantile(0.75)+1.5*IQR

# so make it is 0

#######################################################

# Trimming
import numpy as np

outliers_df=np.where(df.black>upper_limit,True,np.where(df.black<lower_limit,True,False))
# you can check outliers_df column in variable explorer
df_trimmed=df.loc[~outliers_df]
df.shape
# 506,15
df_trimmed.shape
# 429,15

########################################################

# Replacement technique ----> masking
# Drowback of trimming technique is we are losing the data

df.describe()


df_replaced=pd.DataFrame(np.where(df.black>upper_limit,upper_limit,np.where(df.black<lower_limit,lower_limit,df.black)))
# if the values greater than Upper_limit
# map it to upper limit and less than lower limit
# map it to lower limit , if it is within the range
# then keep as it is

sns.boxplot(df_replaced[0])

#########################################################
# Winsorizer

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['black']
                  )

# Copy Winsorizer and paste in Help tab of
# top right window, study the method

df_t=winsor.fit_transform(df[['black']])

sns.boxplot(df[['black']])
sns.boxplot(df_t['black'])


##############################################################
