import pandas as pd
from sklearn import linear_model

from sklearn.linear_model import LinearRegression

data=pd.read_csv("wine-quality.csv")
print(data)

from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
X = df[list(df.columns)[:-1]]
y=df['quality']
X_train, X_test,y_train,y_test=train_test_split(X,y)
regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_prediction=regressor.predict(X_test)
print('R-score is %s'%regressor.score(X_test,y_test))