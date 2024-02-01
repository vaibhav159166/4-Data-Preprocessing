"""
Created on Tue Oct 10 08:21:38 2023

@author: Vaibhav Bhorkade

"""
import pandas as pd

data=pd.read_csv("C:/datasets/ethnic_diversity.csv")

data.head(10)

data.info()

# It gives size, null values,rows,columns and columns data

data.describe()


data['Salaries_new']=pd.cut(data['Salaries'], bins=[min(data.Salaries),data.Salaries.mean(),max(data.Salaries)],labels=['low','High'])
data.Salaries_new.value_counts()

data['Salaries_new']=pd.cut(data['Salaries'], bins=[min(data.Salaries),data.Salaries.quantile(0.25),data.Salaries.mean(),data.Salaries.quantile(0.75),max(data.Salaries)],labels=['group1','group2','group3','group4'])
data.Salaries_new.value_counts()

###############################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("C:/datasets/animal_category.csv")

df.shape

df.drop(['Index'],axis=1,inplace=True)

df_new=pd.get_dummies(df)

df_new.shape

df_new.drop(['Gender_Male','Homly_Yes'],axis=1,inplace=True)

df_new.shape
# Now we get 30,12
df_new.rename(columns={'Gender_Female':'Gender','Homly_No':'Homly'})

#######################################################################

df=pd.read_csv("C:/datasets/ethnic_diversity.csv")

df.shape

df.head()

# df.drop(['Index'],axis=1,inplace=True)

df_new=pd.get_dummies(df)

df_new.shape

df_new.drop(['EmpID','Zip','Salaries','age'],axis=1,inplace=True)

df_new.shape

######################################

# one hot encoder

import pandas as pd

from sklearn.preprocessing import OneHotEncoder

enc=OneHotEncoder()

df=pd.read_csv("C:/datasets/ethnic_diversity.csv")

df.columns
# we have Salries and age as numerical columns, let us make then
# at the position 0 and 1 so that make further data processing easy
df=df[['Salaries','age','Employee_Name', 'Position', 'State', 'Sex',
       'MaritalDesc', 'CitizenDesc', 'EmploymentStatus', 'Department', 'Race']]
# Check the dataframe in variable explorer
# we want only nominal data and ordinal data for processing
# hence skipped 0 th and first column and applied to one hot encoder

enc_df=pd.DataFrame(enc.fit_transform(df.iloc[:,2:]).toarray())

#######################################################