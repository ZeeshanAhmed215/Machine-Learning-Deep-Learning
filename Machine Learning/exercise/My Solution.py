# Step 1=========================

import pandas as pd
import matplotlib.pyplot as plt
from pyparsing import line
from sklearn.linear_model import LinearRegression
df=pd.read_csv("E:\Online Courses\Programs\ML & DL\Machine Learning\exercise\salary_data.csv")
print(df.head())
# Step 2=========================
print(df.describe())
years_exp=list(df["years_experience"])
salary=list(df["salary_lpa"])

plt.scatter(years_exp,salary)
plt.xlabel("Years of Experience")
plt.ylabel("Salary LPA")
plt.title("Employees Experience and Salary")
plt.show()

# Step 3=========================
X=df[["years_experience","education_level","certifications"]]
y=df["salary_lpa"]
model1=LinearRegression()
model1.fit(X,y)
cof=model1.coef_
inter=model1.intercept_
sal=cof[0]*4+cof[1]*2+cof[2]*3+inter

# Step 4=========================
df["predicted_salary"]=model1.predict(X)
p=model1.predict([[4,2,3]])
print(p[0])
print(df.head())