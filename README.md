# APP_DS_EX1
Implementing Data Preprocessing and Data Analysis

## Aim:
To implement Data analysis and data preprocessing using a data set

## Algorithm:
Step 1: Import the data set necessary

Step 2: Perform Data Cleaning process by analyzing the sum of Null values in each dataset column.

Step 3: Perform Categorical data analysis.

Step 4: Use Sklearn tool from python to perform data preprocessing such as encoding and scaling.

Step 5: Implement Quantile transfomer to make the column value more normalized.

Step 6: Analyzing the dataset using visualizing tools from matplot library or seaborn.

## Program and Output:
```
Developed by:   Yashwanth Raja Durai V
Register no: 212222040184
```
#### Importing Libraries:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from category_encoders import BinaryEncoder
from sklearn.preprocessing import MinMaxScaler,RobustScaler
df=pd.read_csv('Life Expectancy Data CSV.csv')
```
```python
df.head()
```
![image](https://github.com/user-attachments/assets/33949cbc-34a1-42b9-b2da-aa80c90982c0)
```python
df.info()
```
![image](https://github.com/user-attachments/assets/307b286d-8cc8-4fea-adc0-24defeaeb57d)

```python
df.isnull().sum()
```
![image](https://github.com/user-attachments/assets/590c3c80-5907-4782-ad95-5c0522aad5c7)

```python
numerical_columns=df.select_dtypes(include=['number']).columns
numerical_columns
```
![image](https://github.com/user-attachments/assets/d4d9ef12-2cf3-42e9-9682-9fc28094bf49)

#### Data Cleaning
```python
columns_to_fill = [
    'Life expectancy ', 'Adult Mortality', 'Alcohol', 'Hepatitis B',
    ' BMI ', 'Polio', 'Total expenditure', 'Diphtheria ',
    'GDP', 'Population', 'Income composition of resources', 'Schooling'
]
df[columns_to_fill] = df[columns_to_fill].fillna(df[columns_to_fill].median())

df.isnull().sum()
```
![image](https://github.com/user-attachments/assets/8600c495-5ae9-4da3-a9eb-665b27bc2e02)

#### Before Removing Outliers

```python
numerical_columns = ['Life expectancy ', 'Adult Mortality', 'infant deaths', 'Alcohol', 
                     'percentage expenditure', 'Hepatitis B', 'Measles ', ' BMI ', 
                     'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ', 
                     'GDP', 'Population', 'Income composition of resources', 'Schooling']

fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 12))  # Adjust rows/cols based on number of features
axes = axes.flatten()

for i, column in enumerate(numerical_columns):
    sns.boxplot(data=df, x=column, ax=axes[i])
    axes[i].set_title(column)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/3802cc12-b288-4336-b04b-f33d61859b6b)

#### Removing Outliers using IQR
```python
df_cleaned = df.copy()

for column in numerical_columns:
    Q1 = df_cleaned[column].quantile(0.25)
    Q3 = df_cleaned[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_cleaned = df_cleaned[(df_cleaned[column] >= lower_bound) & (df_cleaned[column] <= upper_bound)]

print("Shape of DataFrame after outlier removal:", df_cleaned.shape)
```
![image](https://github.com/user-attachments/assets/21091dd5-e2b7-4d6b-898f-a4086bd4514b)

#### After Removing Outliers
```python
for i, column in enumerate(numerical_columns):
    sns.boxplot(data=df_cleaned, x=column, ax=axes[i])
    axes[i].set_title(column)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

```
![image](https://github.com/user-attachments/assets/7c41e49e-af46-4033-8c8c-27197e81ad44)

#### Identifying the categorical data and performing categorical analysis.
```python
categorical_columns = df_cleaned.select_dtypes(include=['object']).columns
print("Categorical Columns:", categorical_columns)

for column in categorical_columns:
    print(f"Value counts for {column}:")
    print(df_cleaned[column].value_counts())
    print("\n")
```
![image](https://github.com/user-attachments/assets/365afd54-e10c-4a94-8f8c-c79aac22509a)

![image](https://github.com/user-attachments/assets/1e510411-3ee7-4279-a702-f3a2e64ae5e4)

#### Bivariate and multivariate analysis
```python
plt.figure(figsize=(8, 6))
sns.scatterplot(x='GDP', y='Life expectancy ', data=df_cleaned)
plt.title('Scatter Plot: Life expectancy vs GDP')
plt.show()
```
![image](https://github.com/user-attachments/assets/b4937960-b2f9-46de-9cef-914d1b67e557)

```python
plt.figure(figsize=(8, 6))
sns.countplot(x='Year', hue='Status', data=df_cleaned)
plt.title('Count Plot: Year vs Status')
plt.xticks(rotation=90)
plt.show()
```
![image](https://github.com/user-attachments/assets/3675c88f-eb27-4097-85fe-49222b86d024)

```python
# Grouped box plot: 'Life expectancy' by 'Status' and 'Year'
plt.figure(figsize=(10, 6))
sns.barplot(x='Year', y='Life expectancy ', hue='Status', data=df_cleaned)
plt.title('Life Expectancy by Year and Status')
plt.xticks(rotation=90)
plt.show()
```
![image](https://github.com/user-attachments/assets/14bd8163-70a3-4cbf-8220-c2f9dd268925)

#### Data Encoding
```python
le=LabelEncoder()
df_cleaned['Country']=le.fit_transform(df_cleaned['Country'])

be=BinaryEncoder()
nbe=be.fit_transform(df_cleaned['Status'])
df_cleaned=pd.concat([df_cleaned,nbe],axis=1)
df_cleaned.drop(columns=['Status'],inplace=True)
```

#### Data Scaling
```python
scaler=MinMaxScaler()
columns_to_scale=['Life expectancy ','infant deaths', 'Alcohol', 'Hepatitis B',
       ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure',
       'Diphtheria ', 'GDP', 'Income composition of resources',
       'Schooling']
df_cleaned[columns_to_scale]=scaler.fit_transform(df_cleaned[columns_to_scale])

rscaler=RobustScaler()
columns_to_rscaler=['Adult Mortality', 'percentage expenditure','Measles ', 'Population']
df_cleaned[columns_to_rscaler]=rscaler.fit_transform(df_cleaned[columns_to_rscaler])
df_cleaned.head()
```
![image](https://github.com/user-attachments/assets/51e0cf58-26ee-47b3-9731-730e7b53f924)

#### Data Visualization
#### HeatMap
```python
corr_matrix = df_cleaned.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
```
![image](https://github.com/user-attachments/assets/f19ef3b6-7ada-477c-a20c-40ceb558ee2c)

#### Pairplot
```python
selected_columns = ['Life expectancy ', 'GDP', 'Alcohol', ' BMI ', 'Schooling']
sns.pairplot(df_cleaned[selected_columns])
plt.suptitle('Pairplot of Selected Numerical Columns', y=1.02)
plt.show()
```
![image](https://github.com/user-attachments/assets/5a8fa4d8-7c80-4307-ab51-c0d5b23fe646)


## Result:
Thus Data analysis and Data preprocessing are implemented using a dataset.
