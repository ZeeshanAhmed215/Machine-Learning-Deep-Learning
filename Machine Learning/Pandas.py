
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df=pd.read_csv("E:\Online Courses\Programs\ML & DL\Machine Learning\student_exam_performance_cleaned.csv")

df['time_management_score']=df["time_management_score"].fillna(df['time_management_score'].mean())
df['study_hours_per_day']=df["study_hours_per_day"].fillna(df['study_hours_per_day'].mean())
df['sleep_hours']=df["sleep_hours"].fillna(df['sleep_hours'].mean())
df['exam_anxiety_level']=df["exam_anxiety_level"].fillna(df['exam_anxiety_level'].mean())
df['exam_score']=df["exam_score"].fillna(df['exam_score'].mean())

features=df[['time_management_score','study_hours_per_day','sleep_hours','exam_anxiety_level']]
target=df['exam_score']

# plt.scatter(df['study_hours_per_day'],df['exam_score'])
# plt.xlabel("Study Hours Per Day")
# plt.ylabel("Exam Score")
# plt.title("Study Hours Per Day vs Exam Score")
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
# print(model.coef_)
# print(model.intercept_)
df["predicted_exam_score"]=model.predict(features)
# print(df[["exam_score","predicted_exam_score"]])

accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy   *100, "%")