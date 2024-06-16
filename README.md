# Diabetes Progression Prediction

## Description
This [project](https://github.com/AnalysisWithBrett/CodingTasks/blob/main/diabetes_regression.ipynb) aims to predict diabetes progression using various features from the dataset [diabetes_dirty.csv](https://github.com/AnalysisWithBrett/CodingTasks/blob/main/diabetes_dirty.csv). The task involves data cleaning, exploratory data analysis (EDA), feature scaling, and building a linear regression model to make predictions. Understanding this aspect of coding is crucial for developing skills in data preprocessing, visualization, and machine learning model implementation.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Data Import and Initial Analysis](#data-import-and-initial-analysis)
  - [Data Preprocessing](#data-preprocessing)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Feature Scaling](#feature-scaling)
  - [Model Training and Evaluation](#model-training-and-evaluation)
- [Screenshots](#screenshots)
- [Credits](#credits)

## Installation
To run this project locally, you need to have Python installed along with the following libraries:
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn

You can install these libraries using pip:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```
## Usage
### Data Import and Initial Analysis
First, we import the necessary libraries and the dataset. We then perform initial data exploration to understand its structure and content.
```bash
# Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Importing the dataset
df = pd.read_csv("diabetes_dirty.csv")
df.sample(10)
```
###### This samples the data randomly where you can also see the values.
![Sample](https://github.com/AnalysisWithBrett/CodingTasks/blob/main/samples.png)
```bash
# Display the dimensions of the dataset
print(df.shape)
```
###### This tells you the dimensions of the dataset.
![Shape](https://github.com/AnalysisWithBrett/CodingTasks/blob/main/shape.png)
```bash
# Display the information about the dataset
print(df.info())
```
###### This provides a concise overview of the DataFrame's size, data types, and non-null values, helping to quickly understand its structure and potential issues like missing data.
![Info](https://github.com/AnalysisWithBrett/CodingTasks/blob/main/info.png)
```bash
# Check for any missing values
print(df.isnull().sum())
```
###### This displays the number of nulls in each variable.
![Nulls](https://github.com/AnalysisWithBrett/CodingTasks/blob/main/nulls.png)
## Data Preprocessing
### We split the dataset into features (X) and target variable (y), then further split these into training and test sets.
```bash
# Splitting the dataset into features and target variable
X = df.drop("PROGRESSION", axis=1)
y = df['PROGRESSION']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
```
###### This shows the dimensions for each of the subset you created.
![Train Test](https://github.com/AnalysisWithBrett/CodingTasks/blob/main/traintest%20shape.png)
## Exploratory Data Analysis
### We create visualizations to understand the relationships between variables.
```bash
# Creating a heatmap to see the correlation between the features
sns.heatmap(df.corr(), annot=True, cmap='YlGnBu')
```
###### This generates a heatmap with annotations to show the correlations between the variables.
![heatmap](https://github.com/AnalysisWithBrett/CodingTasks/blob/main/heatmap.png)
```bash
# Creating a pairplot to see the correlation and the distribution of the features
sns.pairplot(X)
```
###### This creates a grid of scatterplots for visualizing pairwise relationships between variables.
![pairplot](https://github.com/AnalysisWithBrett/CodingTasks/blob/main/pairplot.png)
```bash
# Using a loop to plot histograms to see the distribution of the features
for i in X.columns:
    plt.figure(figsize=(3,3))
    sns.histplot(X[i], kde=True, color="blue")
    plt.show()
```
###### This creates histogram to show the distribution of the data. I only showed three here for simplification.
![histogram1](https://github.com/AnalysisWithBrett/CodingTasks/blob/main/hist1.png)
![histogram2](https://github.com/AnalysisWithBrett/CodingTasks/blob/main/hist2.png)
![histogram3](https://github.com/AnalysisWithBrett/CodingTasks/blob/main/hist3.png)
```bash
## Feature Scaling
### Standardising the features to have a mean of 0 and a standard deviation of 1.
```bash
# Initializing the standard scaler
sc = StandardScaler()

# Fitting the standard scaler with the training data
sc.fit(X_train)

# Applying the scaler on train and test data
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)
```
## Model Training and Evaluation
### Training a linear regression model and evaluating its performance.
```bash
# Initializing the linear regression model
lm = LinearRegression()

# Fitting the model with the training data
model = lm.fit(X_train, y_train)

# Displaying the intercept and coefficients of the model
print(f"Intercept: {lm.intercept_}")
print(f"Coefficients: {lm.coef_}")
```
###### This prints the intercept and coefficients of a linear regression model.
![Intercepts Coefficients](https://github.com/AnalysisWithBrett/CodingTasks/blob/main/intercept%20coefficietns.png)
```bash

# Making predictions using the fitted linear model
lm_pred = lm.predict(X_test)

# Calculating and displaying the R-squared score
r_sq = r2_score(y_test, lm_pred)
print(f"R-squared: {r_sq}")
```
###### This prints the R-square, which measures the goodness of fit of a regression model and indicates how well the model fits the observed data.
![R-sq](https://github.com/AnalysisWithBrett/CodingTasks/blob/main/r-sq.png)
## Credits
This project is developed by [Brett Hoy](https://github.com/AnalysisWithBrett). If you have any questions or suggestions, feel free to contact me.
