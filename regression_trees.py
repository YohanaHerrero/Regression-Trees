import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

#load data
data = pd.read_csv("real_estate_data.csv")
data.head()

#is the dataset complete?
data.isna().sum()
#some rows have several NA: crim, zn, indus, chas, age, lstat

#data preprocessing
#drop the rows with missing values because we have enough data in our dataset
data.dropna(inplace=True)
data.isna().sum()

#split the dataset into attributes and target
X = data.drop(columns=["MEDV"])
y = data["MEDV"]

#split dataset into training and test
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=.2, random_state=1)

#create regression tree
regression_tree = DecisionTreeRegressor(criterion = 'squared_error') #criterion: The function used to measure error

#train the model
regression_tree.fit(X_train, Y_train)

#make predictions
prediction = regression_tree.predict(X_test)

#evaluate model accuracy
#I use the score method = R2, which indicates the coefficient of determination
regression_tree.score(X_test, Y_test)

#find the average error in our testing set = average error in median home value prediction
print((prediction - Y_test).abs().mean()*1000,"$")

#we can try to improve the accuracy by changing the regression tree model
#we can use the criterion absolute_error instead of squared_error
regression_tree2 = DecisionTreeRegressor(criterion = 'absolute_error')
regression_tree2.fit(X_train, Y_train)
prediction2 = regression_tree2.predict(X_test)
print(regression_tree2.score(X_test, Y_test))
print((prediction2 - Y_test).abs().mean()*1000,"$")
