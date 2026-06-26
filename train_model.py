import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ml models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.linear_model import LogisticRegression# Logistic regression
from sklearn.neighbors import KNeighborsClassifier # K-nn
from sklearn.naive_bayes import GaussianNB # Naive bayes
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier

# splitting test_train
from sklearn.model_selection import train_test_split
#cross validation
from sklearn.model_selection import KFold, cross_val_score
# evaluation metrics
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

"""LOAD DATA SET"""

data = pd.read_csv("beer-servings.csv")

#from google.colab import drive
#drive.mount('/content/drive')

data.head()

data.describe()

print(data.shape)
print(data.info())
print(data.describe())

data.isnull().sum()

data.duplicated().sum()

#data.drop_duplicates(inplace = True)

missing_percent = data.isna().sum()/ len(data)* 100

print(missing_percent)

num_cols = data.select_dtypes(include =["number"]).columns
cat_cols = data.select_dtypes(include =["object"]).columns

num_cols

cat_cols

plt.figure(figsize=(15,10))
for i,col in enumerate(num_cols, 1):
  plt.subplot(4,2,i)
  plt.hist(data[col])
  plt.title(col, fontsize=10)
  plt.title(col)

plt.show()

# plots to understand categorical value distribution
plt.figure(figsize= (15,10))
for i,col in enumerate(cat_cols, 1):
  #plt.subplot(2, 2, i)
  plt.subplot(2,1, i)
  data[col].value_counts().plot(kind ='bar')
  plt.title(col)
  plt.xticks(rotation = 45)
plt.tight_layout
plt.show()

"""Missing value handling
1.Remove
1.1 Remove rows: if a count of missing value is more than 30% of the no. of rows, we remove coulmn itself.

1.1 Remove columns: when we know that removal of a row doesnt make much difference in terms of available data for ML model building

2.Replace

2.1 Mode: if the colums has non numerical data

2.2 Mean : if that data distribution of the coulmn is symmetrical(normal distribution)

2.3 Median: if that data distribution of the coulmn is not symmetrical(skwed distribution)
"""

data.fillna(data.median(numeric_only=True), inplace=True)

#for mode:
#data['column name'].fillna(
    #data['column'].mode()[0],
    #inplace=True
#)

data.isna().sum()

#removed unwanted columns
data.drop(
    ['Unnamed: 0','country'],
    axis=1,
    errors='ignore',
    inplace=True
)

"""##EDA"""

data.hist(figsize=(10,8))
plt.show()

"""##OUTLIERS

quartile and IQR plays a role in identifying outliers
Q1 = 25% of the column (25th percentile)

Q2 = 50% of the column (55th percentile)

Q3 = 75% of the column (75th percentile)

Range = Max -Min

IQR = Q3-Q1

IQR :Inter Quartile Range

Upper limit = Q3 + (1.5 * IQR)

Lower Limit = Q1 -(1.5 * IQR)

Outerlier : any value > upper limit or value < lower limit we handle outliers bcoz they can skew the general nature of the ML model
"""

data.describe()

print(num_cols)

data.drop(
    ['Unnamed: 0','country'],
    axis=1,
    errors='ignore',
    inplace=True
)

print(data.columns.tolist())
print(num_cols)

num_cols = data.select_dtypes(include=np.number).columns
print(num_cols)

plt.figure(figsize=(10,10))

for i, col in enumerate(num_cols, 1):
    plt.subplot(2, 3, i)
    plt.boxplot(data[col])
    plt.title(col)
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

#outlier handling can be done in 2 ways
##Identify outliers
#Values below lower limit → outliers
#Values above upper limit → outliers
# 1.removing all rows where outliers are present
# 2. clipping the values to upper limit and lower limit

Q1 = data[num_cols].quantile(0.25)

Q3 = data[num_cols].quantile(0.75)
IQR = Q3 - Q1
upper_limit = Q3+ (1.5 * IQR)
lower_limit = Q1- (1.5 * IQR)

outliers = ((data[num_cols] > upper_limit)|(data[num_cols] < lower_limit)).any(axis = 1)

outliers_df = data[outliers]
outlier_percentage = len(outliers_df)* 100/len(data)

print('percentage of outliers in the dataframe: ', outlier_percentage)

#option 2 : clipping
#customer_df[num_cols] = customer_df[num_cols].clip(lower_limit, upper_limit, axis=1)
# axis =1 means along with column direction

data[num_cols] = data[num_cols].clip(lower_limit, upper_limit, axis=1)

#data['wine_servings'] = data['wine_servings'].clip(
    #lower=lower_limit,
    #upper=upper_limit
#)

#data['spirit_servings'] = data['spirit_servings'].clip(
    #lower=lower_limit,
    #upper=upper_limit
#)

plt.figure(figsize=(10,10))

for i, col in enumerate(num_cols, 1):
    plt.subplot(2, 3, i)
    plt.boxplot(data[col])
    plt.title(col)
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

"""#Encoding
label_encoder = LabelEncoder()

customer_df["city"] = label_encoder.fit_transform(customer_df["city"])

customer_df["gender"] = label_encoder.fit_transform(customer_df["gender"])

doing one hot encoding on mentioned coulmns
customer_df = pd.get_dummies(customer_df, columns =['education_level'],dtype =int)
"""

data = pd.get_dummies(
    data,
    columns=['continent'],
    drop_first=True
)

data.head()

# Separate features (X) and target variable (y)
X = data.drop(columns=['total_litres_of_pure_alcohol'])
y = data['total_litres_of_pure_alcohol']

# Split the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Initialize the models
lr_model = LinearRegression()
ridge_model = Ridge()

# ----------------- Linear Regression -----------------
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

print("--- Linear Regression Performance ---")
print(f"MSE: {mean_squared_error(y_test, lr_preds):.4f}")
print(f"R2 Score: {r2_score(y_test, lr_preds):.4f}\n")

# ----------------- Ridge Regression -----------------
ridge_model.fit(X_train, y_train)
ridge_preds = ridge_model.predict(X_test)

print("--- Ridge Regression Performance ---")
print(f"MSE: {mean_squared_error(y_test, ridge_preds):.4f}")
print(f"R2 Score: {r2_score(y_test, ridge_preds):.4f}")

from sklearn.linear_model import LinearRegression
import joblib

# Train the model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Save the model
joblib.dump(lr_model, "beer_prediction_model.pkl")

print("Model saved successfully!")

import joblib

model = joblib.load("beer_prediction_model.pkl")
