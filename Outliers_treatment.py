# Outliers treatment
import pandas as pd
import seaborn as sns

df=pd.read_csv("C:/datasets/ethnic_diversity.csv")

sns.boxplot(df.Salaries)
# There are outliers

sns.boxplot(df.age)
# There is no outliers

IQR=df.Salaries.quantile(0.75)-df.Salaries.quantile(0.25)
# Have observed IQR in variable explorer
# no,because IQR is in capital letters
# treated as constant
                       
IQR
# but if we try as I,Iqr, or iqr then it is showing

lower_limit=df.Salaries.quantile(0.25)-1.5*IQR

upper_limit=df.Salaries.quantile(0.75)+1.5*IQR

# salary, it is -19446
# So there is -ve salary
# so make it is 0

#######################################################

# Trimming
import numpy as np

outliers_df=np.where(df.Salaries>upper_limit,True,np.where(df.Salaries<lower_limit,True,False))
# you can check outliers_df column in variable explorer
df_trimmed=df.loc[~outliers_df]
df.shape
# 310,13
df_trimmed.shape
# 306,13

########################################################

# Replacement technique ----> masking
# Drowback of trimming technique is we are losing the data

df=pd.read_csv("C:/datasets/ethnic_diversity.csv")

df.describe()

# record no. 23 has got outliers
# map all the outliers

df_replaced=pd.DataFrame(np.where(df.Salaries>upper_limit,upper_limit,np.where(df.Salaries<lower_limit,lower_limit,df.Salaries)))
# if the values greater than Upper_limit
# map it to upper limit and less than lower limit
# map it to lower limit , if it is within the range
# then keep as it is

sns.boxplot(df_replaced[0])

#########################################################
"""
Created on Fri Oct  6 08:19:26 2023

@author: Vaibhav Bhorkade

"""
# Winsorizer

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['Salaries']
                  )

# Copy Winsorizer and paste in Help tab of
# top right window, study the method

df_t=winsor.fit_transform(df[['Salaries']])

sns.boxplot(df[['Salaries']])
sns.boxplot(df_t['Salaries'])