"""
Created on Thu Oct  5 09:31:11 2023

@author: Vaibhav Bhorkade

"""

import pandas as pd

df=pd.read_csv("C:/datasets/ethnic_diversity.csv")
print(df)

df.shape

df.columns

df.dtypes

# Lets convert the data types into float

df.Salaries=df.Salaries.astype(int)
df.dtypes
# Now salries is int

# Presentaly it is in int
# Similaraly age data must be float
df.age=df.age.astype(float)
df.dtypes


##########################################

# Identify the duplicates
 
df_new=pd.read_csv("C:/datasets/education.csv")
duplicate=df_new.duplicated()
# Output of this function is single columns
# if there is duplicate records output- True
# if there is no duplicate records output-False
# Series will be created

duplicate
sum(duplicate)
# output will be zero

# Another dataset

df_new1=pd.read_csv("C:/datasets/mtcars_dup.csv")
duplicate1=df_new1.duplicated()
# Output of this function is single columns
# if there is duplicate records output- True
# if there is no duplicate records output-False
# Series will be created
duplicate1
# There are 3 duplicate records
# row 17 is duplicate of row 2 like wise you can 3 duplicate

sum(duplicate1)
# sum is 3

df_new2=df_new1.drop_duplicates()
duplicate2=df_new2.duplicated()
duplicate2

sum(duplicate2)
