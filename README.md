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
The above code snippet creates a grid of bar plots where each plot represents the distribution of a categorical feature in the dataset. figsize, title, xticks same as the above in terms of functionality, index=1 is to keep track of the subplot position in the grid which increases by 1 with every iteration...represents each plot's position in the grid, value_counts() returns the occurrence of each unique value in the column, and , .subplot() creates a subplot within the grid, the parameters of the subplot specifies the dimensions of the plot, which is 11 units tall and 4 units wide. The sns.barplot is a bar plot made using seaborn, the parameters are...the x-values are the unique values of the categorical values, and the y-values are the counts of each value. <br>
<br>

```
dataset.drop("Id", axis = 1, inplace = True)
```
This line drops the entire column named "Id" from the dataset. .drop is a method in pandas used to drop a specified row or a column from a dataset. axis=1, specifies to the .drop method that a column needs to be dropped, inplace=True specifies that the dataset needs to be modified after the dropping takes place, if inplace=False or if its not specified, then a new dataset is created without altering the original dataset. <br>
<br>

```
dataset["SalePrice"] = dataset["SalePrice"].fillna(dataset["SalePrice"].mean())
```
This line modifies the "SalePrice" column, by using the .fillna pandas method on the column to fill in the missing values with the mean values of the column by using the .mean method on the dataset column. <br>
<br>

```
new_dataset = dataset.dropna()
```
This line creates a new dataset called "new_dataset", which takes a reference of the original dataset and uses the .dropna() pandas method, which drops all the rows that contain missing values in them. <br>
<br>

```
new_dataset.isnull().sum()
```
This above line checks if there are any remaining missing values in this newly created dataset from the original dataset by using the .isnull method, which checks for any null or missing values in the columns of the dataset, and then the .sum() calculates the total number of the missing values if present in the dataset's columns. <br>
<br>

```
s = (new_dataset.dtypes == "object")
object_cols = list(s[s].index)
print(f"Categorical variables: {object_cols}")
print(f"No. of categorical variables: {len(object_cols)}")
```
This code snippet is used to identify the categorical values in the dataset, and print their names along with the total count. The first line creates a series of values where each value represents whether the corresponding column is a categorical value(object) or not. The second line is used to extract the column names from the above series where the corresponding element is True, these values are stored in the object_cols list. The remaining two lines are used to print the values of the object_cols and the total count of the list containing the values.<br>
<br>

```
OH_encoder = OneHotEncoder(sparse_output=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols]))
OH_cols.index = new_dataset.index
OH_cols.columns = OH_encoder.get_feature_names_out()
df_final = new_dataset.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_cols], axis=1)
```
The above code snippet performs one hot encoding on the categorical variables in the new_dataset dataframe, and creates a new dataset dataframe, and creates a new dataframe called df_final with the encoded features. The first line uses the OneHotEncoder class and creates an instance of it and uses the sparse_output parameter set to False which means the encoded output will be in an array format. The second line uses the pandas DataFrame method creation, and applies one hot encoding to the categorical variables in the new_dataset[object_cols] and stores the result in a new dataframe created in this line in the OH_cols variable. The next line sets the index of the OH_cols dataframe to match the index of the new_dataset dataframe. The next line, OH_cols.columns assigns appropriate column names to the one hot encoded features in the OH_cols dataframe using the OH_encoder.get_feature_names_out() method of the OneHotEncoder class. The next line, df_final creates a new dataframe created which references the new_dataset variable and drops the original categorical columns before one hot encoding. The last line uses the pandas method .concat to merge the df_final dataframe and the OH_cols dataframe in place along the columns. <br>
<br>

```
X = df_final.drop(["SalePrice"], axis = 1)
Y = df_final["SalePrice"]
```
In the above snippet, X creates a feature matrix and uses the df_final dataframe as a reference and drops the "SalePrice" column. Y creates a target variable by assigning its value to that of the df_final["SalePrice"] column and its values. <br>
<br>

```

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, train_size = 0.8, test_size = 0.2, random_state = 0)
```
The line stands uses the train_test_split function from scikit-learn and then is used to split the dataset into the training and the validation sets. X_train and X_valid stand for the respective training features and the validation features which are not included in the training features matrix. The Y_train and the Y_valid represent the target variable corresponding to the training and the validation sets. The train_size with the parameter set to 0.8 divides the dataset distribution to 80% given to the training sets (X and Y), and the remaining part indicated part by the test_size goes to the cross validation set (X and Y). The random_state value set to 0, ensures reproducibility of the random split to allow consistent and reproducibile results. <br>
<br>

```
model_SVR = svm.SVR()
model_SVR.fit(X_train, Y_train)
Y_pred = model_SVR.predict(X_valid)
mean_absolute_percentage_error(Y_valid, Y_pred)
```
The above code snippet, creates an instance of the Support vector regression model by using the svm.SVR class from the scikit learn library. Then the model_SVR is trained on the X_train and the Y_train datasets. This basically involves finding the optimal hyperplane that best fits the training data. The next line uses the .predict method which takes the input of the x_valid matrix and returns the predicted values Y_pred based on the learning of the model. The mean_absolute_percentage_error(MAPE) is calculated by comparing the predicted target values and Y_pred and the actual target_values Y_valid. MAPE is a used evaluation metric to find the average percentage difference between the predicted and the actual values. MAPE can help assess the performance of the SVR model on the validation data. <br>
<br>

```
model_RFR = RandomForestRegressor(n_estimators = 10)
model_RFR.fit(X_train, Y_train)
Y_pred = model_RFR.predict(X_valid)
mean_absolute_percentage_error(Y_valid, Y_pred)
```
This code snippet is quite similar to the previous one but is a but different in the functionality. The first line creates an instance of the RandomForestRegressor class stored in the model_RFR variable. RandomForestRegressor is an ensemble tree learning method, and it uses the n_estimators which specifies the number of decision trees to be included in the random forest. The second line trains the RandomForestRegressor model on the training data (X_train), and the target variables (Y_train). The model learns the patterns and the relationships in the training data (X_train and Y_train) in order to make accurate predictions. The next line, uses the trained forest regressor model to make predictions on the validation data (X_valid), this model applies the learned patterns and the relationships on the X_valid matrix and returns the predictions made. In the next line as in the previous code snippet, MAPE is used to calculate the mean_absolute_percentage_error, the percentage difference between the predicted and the actual values of the target variables/values. <br>
<br>

```
model_LR = LinearRegression()
model_LR.fit(X_train, Y_train)
Y_pred = model_LR.predict(X_valid)
mean_absolute_percentage_error(Y_valid, Y_pred)
```
The above code snippet performs linear regression on the model. The first line creates an instance of the LinearRegression class. LinearRegression assumes a linear relationship between the features/independent variables and the dependent variables/target variables. Once this is done, in the next class, the LinearRegression model instance is trained on the X_train and the Y_train variables in order to understand the patterns and the relationships between the variables and their target values/variables. The model estimates the coefficients and find the best line fit to the training data, not overfit or underfit but the best fit. This learned/trained model then makes predictions on the X_valid validation matrix. The next line like the previous code snippets uses MAPE to calculate the mean_absolute_percentage_error, this is used to find the difference between the predicted and the actual values of the trained model.