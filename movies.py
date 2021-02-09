import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns

plt.style.use('seaborn-whitegrid')

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder


# import seaborn as sns

# reads csv file
movies = pd.read_csv("MoviesOnStreamingPlatforms_updated.csv")
movies = movies.drop("Unnamed: 0",axis=1)

# the top 10
print(movies.head(10))

# summary of the DataSet
print(movies.info())

print(movies['Genres'].value_counts())

df_movies = movies

#Removing the Target value
df_movies = df_movies[df_movies['IMDb'].notna()]

#Visualizing the amount of missing data
msno.bar(df_movies ,color='red', figsize=(10, 4))


#Dropping all the rows(entries) where there are celss with no data
df_movies.dropna(axis=0, how='any',inplace=True)

df_movies.isna().sum()

#Visualizing if there is anymore missing data or not
print(msno.bar(movies ,color='red', figsize=(10, 4)))

# Choose target and features
y = df_movies.IMDb

X = df_movies.drop(['IMDb'], axis=1)

#Split the data for train and test
X_train_full, X_test_full, y_train, y_test  = train_test_split(X, y,random_state = 0)

#List of Categorical colunmns to be used as features
cat_cols=["Age","Directors","Genres","Country","Language"]

#List of Numerical colunmns to be used as features
numerical_cols = ['Year','Runtime']

#Keep selected columns only
my_cols = numerical_cols + cat_cols
X_train = X_train_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()