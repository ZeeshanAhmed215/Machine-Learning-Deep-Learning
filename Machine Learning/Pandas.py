# from sklearn.datasets import load_breast_cancer
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# # Load the dataset directly as features (X) and target (y)
# X, y = load_breast_cancer(return_X_y=True)

# # Split into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # print(X_train)
# # Train a basic linear regression model
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)
# prediction=model.predict(X_test)
# # print("Model Coefficients:", model.coef_)
# accuracy=accuracy_score(y_test,prediction)
# print(accuracy)  











import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df=pd.read_csv("E:\Online Courses\Programs\ML & DL\Machine Learning\SalaryGender.csv")
X=[[ 'Gender', 'Age', 'PhD']]
y=['Salary']
print(df.head())

X_train,y_train,X_test,y_test=train_test_split(X,y,random_state=42,test_size=0.2)
model=LinearRegression()
model.fit(X_train,y_train)
predict=model.predict(X_test)
accuracy=model.score(y_test,predict)
print(accuracy)
