import pandas as pd      #imports necessary libraries for data manipulation
import numpy as np   # numerical computations
import matplotlib.pyplot as plt  # visualization
from sklearn.metrics import r2_score #model evaluation

#Loading and Exploring Data
df = pd.read_csv('/content/Height_Weight_Dataset.csv') #reading a .csv file
df

df.head() #Prints the first 6 rows of the dataset

df.shape #prints the shape of the data( in rows & columns)

df.info()  #Prints the info of the Data set

df.isnull().sum() #Prints the Potential Missing values/Null values if any

# x = df['Age'].values
# print(x)
# y = df['Height'].values
# print(y)

#Preparing Data for Modeling

x = df[['Age']].values #Extracts the "Age" column into a NumPy array
y = df[['Height']].values #Extracts the "Height" column into a NumPy array
print(x)
print(y)

import matplotlib.pyplot as plt

# Assuming x and y are defined somewhere
plt.plot(x, y)
plt.xlabel("Age") #Labels the plot axes with "Age" (X-axis) and "Height" (Y-axis)
plt.ylabel("Height")
plt.show()
#Displays the plot

#Splitting Data and Training Linear Regression Model

from sklearn.model_selection import train_test_split #Splits the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state= 42)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train) #Fits the model on the training data
print(lr)

y_pred = lr.predict(x) #predicts the heights for all ages using trained linear regression model
acc =r2_score(y, y_pred)  #R-squared values
print(acc)
plt.scatter(x,y, color='red') #actual data plot
plt.plot(x,y_pred,'g')  #predicted data plot
plt.xlabel('Age')
plt.ylabel('Height')
plt.show()

lr.predict([[91]]) #predicts the height of person based on age in the dataset

"""The result is stored in the variable x_polynom. It will be a NumPy array containing the transformed features, which include the original age and the polynomial terms created based on the degree (age^2, age^3 in this case).
"""

from sklearn.preprocessing import PolynomialFeatures  # imports the PolynomialFeatures
polynom2 = PolynomialFeatures(degree = 3)
x_polynom = polynom2.fit_transform(x)
print(pd.DataFrame(x_polynom).head(50)) #dataset with 50 rows & columns based on degree provided(eg: degree = 3)

from sklearn.model_selection import train_test_split ##Splits the data(x_polynom, y) into training and testing sets
xpoly_train, xpoly_test, ypoly_train, ypoly_test = train_test_split(x_polynom,y, test_size = 0.3)
print(x_polynom.shape)   #prints shape of the dataset
print(y.shape)
print(xpoly_train.shape)
print(xpoly_test.shape)
print(ypoly_train.shape)
print(ypoly_test.shape)

polyreg3 = LinearRegression() #linear regression model polyreg3
polyreg3.fit(xpoly_train, ypoly_train)
print(polyreg3)

ypoly_pred3= polyreg3.predict(x_polynom)
ypoly_pred3 #NumPy array containing the predicted heights for all data points in the original data (df)

"""  ypoly_pred3 is the NumPy array containing the predicted heights for all data points obtained using the polynomial regression model And 'y' containting the actual heights from the DataFrame. r2_score calculates the Rsquared score stored in acc."""

acc = r2_score(ypoly_pred3,y)  #acc is accuracy
acc

plt.scatter(x,y, color = 'red') #Actual data plot
plt.plot(x,ypoly_pred3, color = 'green') #Predicted Data plot
plt.show()

