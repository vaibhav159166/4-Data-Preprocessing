import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

d=pd.read_csv("C:/datasets/mtcars.csv")
d.describe()
a=d.describe()
# Initialize the scalar
scalar=StandardScaler()
df=scalar.fit_transform(d)
dataset=pd.DataFrame(df)
res=dataset.describe()
# here if you will check res , in variable environment then 

##################################################################

# For another dataset
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

d=pd.read_csv("C:/datasets/Seeds_data.csv")
d.describe()
a=d.describe()
# Initialize the scalar
scalar=StandardScaler()
df=scalar.fit_transform(d)
dataset=pd.DataFrame(df)
res=dataset.describe()
# here if you will check res , in variable environment then 

###################################################################

