import pandas as pd

# iris_dirty.csv
iris = pd.read_csv('Datasets/iris_dirty.csv', index_col=0)
iris.columns = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width', 'Species']
iris['Species'] = iris['Species'].str.lower().str.capitalize()
iris.to_csv('Datasets/iris_clean.csv')
print("iris_dirty.csv cleaned -> iris_clean.csv")

# std.csv
std = pd.read_csv('Datasets/std.csv')
std['Notes'] = std['Notes'].replace('yes', 'Yes')
std.loc[std['Listening_in_Class'] == '6', 'Listening_in_Class'] = 'Yes'
std['GPA'] = std['GPA'].replace(0, pd.NA)
std['Weekly_Study_Hours'] = std['Weekly_Study_Hours'].replace(0, pd.NA)
std.to_csv('Datasets/std_clean.csv', index=False)
print("std.csv cleaned -> std_clean.csv")
