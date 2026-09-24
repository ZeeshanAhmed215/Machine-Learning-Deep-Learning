import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


df = pd.read_csv(r"E:\Online Courses\Programs\ML & DL\Machine Learning\House prices\house_prices.csv")


encoded_df = pd.get_dummies(df, columns=['condition', 'waterfront'], drop_first=True, dtype=int)

X = encoded_df.drop(columns=['price',"id","date"]) 
y = encoded_df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

predicted = model.predict(X_test)
my = pd.DataFrame({"Actual": y_test, "Predicted": predicted})

print(my.head())
print(model.score(X_test,y_test)*100)
