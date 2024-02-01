# -*- coding: utf-8 -*-

# Label encoder
# preferaly for nominal data
import pandas as pd

df=pd.read_csv("C:/datasets/ethnic_diversity.csv")
import pandas as pd
from sklearn.preprocessing import LabelEncoder
# creating instance of label
labelencoder=LabelEncoder()
# split your data into input and output variables
x=df.iloc[:,0:9]
y=df['Race']
df.columns
# we have nominal data Sex,MaritalDesc,CitizenDesc
# we want to convert to label encoder
x['Sex']=labelencoder.fit_transform(x['Sex'])
x['MaritalDesc']=labelencoder.fit_transform(x['MaritalDesc'])
x['CitizenDesc']=labelencoder.fit_transform(x['CitizenDesc'])
# label encoder y
y=labelencoder.fit_transform(y)
# This is going to create an array, hence convert
# It is back to dataframe
y=pd.DataFrame(y)
df_new=pd.concat([x,y],axis=1)
# If you will see variables explorer, y do not have column name
# hence the rename column
df_new=df_new.rename(columns={0:'Race'})

########################################################
