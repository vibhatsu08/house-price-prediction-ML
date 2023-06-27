# house-price-prediction-ML
This repository is for the house price prediction project done using linear regression. <br>
Code explanation here ---> <br>
```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
```
The pandas library is imported as is aliased as pd, pandas is a python library used for data manipulation and data analysis in Python. <br>
The matplotlib.pyplot is imported and is aliased as plt, this is a python library used for data visualization. <br>
The seaborn library is imported and this is a library that is built on top of the matplotlib library. It provides more aesthetic options for data visualization. <br>
The OneHotEncoder is imported from the sklearn.preprocessing module, it is a preprocessing technique used to convert the categorical features into numerical values that can be worked and understood by the machine learning algorithms. <br>
The mean_absolute_error and the mean_absolute_percentage_error are imported from the sklearn.metrics module, these are certain evaluation metrics used for calculating the error and the accuracy of a regression model, this works by calculating the absolute differences between the predicted and the actual values. <br>
The train_test_split is imported from the sklearn.model_selection and is used for splitting the dataset into the training and the testing subsets. This is quite important for estimating the machine learning model's performance. <br>
The svm and SVC are imported from the sklearn module, and they represent the Support Vector Machine algorithm and its performance for the implementation of classification tasks. 
RandomForestRegressor is imported from the sklearn.ensemble module, it is a ensemble based machine learning algorithm that constructs multiple decision trees and uses their predictions for making more accurate predictions for regression tasks. <br>
LinearRegression is imported from the sklearn.linear_model and it is used for the application of the linear regression algorithm for modelling the linear relationships between the dependent and the independent variable. <br>
<br>

```dataset = pd.read_excel("/Users/vedantmistry/Downloads/HousePricePrediction.xlsx")
```
This uses the pandas library to import the dataset in the excel format into the notebook. <br>
<br>

```
dataset.head(5)
```
This is used to print the first five rows of the imported dataset. <br>
<br>

```
dataset.shape
```
This prints the overall rows and columns of the dataset, to show the dimensions of the dataset in use. <br>
<br>

```
feature_int = dataset.dtypes == "int"
int_cols = list(feature_int[feature_int].index)
print(f"Integer features/variable: {len(int_cols)})"
feature_float = dataset.dtypes == "float"
float_cols = list(feature_float[feature_float].index)
print(f"Float variables: {len(float_cols)}")
feature_object = dataset.dtypes == "object"
object_cols = list(feature_object[feature_object].index)
print(f"Categorical variables: {len(object_cols)}")
```
The above code mentioned checks if each column in the dataset is of the Int/Float/Object data type, and accordingly creates a Boolean series for the based on each column if it has one of the above datatypes. The .dtypes is a attribute, which provides information about the data types of each column in the dataset. <br>
<br>

```
plt.figure(figsize = (12, 6))
sns.heatmap(dataset.corr(), cmap="BrBG", fmt=".2f", linewidths=2, annot=True)
```
The first line sets the dimensions of the figure, to have a width of 12 units and a height of 6 units. The second line is used for calculating the correlation matrix of the dataset, it measures the linear relationships between the pairs of columns of the dataset, it is used for input data in the heatmap. The cmap stands for the color map, it sets the colors for the map, which in this case is Brown Blue Green, this determines the colors to be used in the heatmap. fmt stands for the format of the numbers to be used in the heatmap, every single number is formatted to be of a floating point data type and be upto two decimal places. The linewidths, sets the width of the lines separating the cells of the heatmap. annot is used for enabling the annotation of the heatmap, where the values of each cell are displayed within each cell. <br>
<br>

```
unique_values = []
for col in object_cols:
    unique_values.append(dataset[col].unique().size)
plt.figure(figsize = (10, 6))
plt.title("No. of unique categorical features/variables")
plt.xticks(rotation = 90)
sns.barplot(x = object_cols, y = unique_values)
```
The purpose of the above code snippet is to find all the unique values in the particular column of the dataset, and then append it to the unique_values array. The for loop basically loops through the object_cols columns, and then the .unique() pandas method findas all the unique values in that column being interated on, which is then appended to the unique_cols array at the top of the code snippet. Once this is done, plt.figure() sets the dimensions of the figure, which is 10 units wide and 6 units tall, the plt.title() sets the title of the plot, the plt.xticks(rotation=90), rotates the x-axis label by 90 degress to avoid overlapping of the labels when there are many categorical features, the sns.barplot(x=object_cols, y=unique_values), uses the seaborn's barplot function, where a bar plot is created that places the values of the object_cols on the x-axis, and the values of the unique_values on the y-axis. <br>
<br>

```
plt.figure(figsize = (18, 36))
plt.title("Categorical features distribution: ")
plt.xticks(rotation = 90)
index = 1
for col in object_cols:
    y = dataset[col].value_counts()
    plt.subplot(11, 4, index)
    plt.xticks(rotation = 90)
    sns.barplot(x = list(y.index), y = y)
    index += 1
```
The above code snippet creates a grid of bar plots where each plot represents the distribution of a categorical feature in the dataset. figsize, title, xticks same as the above in terms of functionality, index=1 is to keep track of the subplot position in the grid, value_counts() returns the occurrence of each unique value in the column 