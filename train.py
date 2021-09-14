from os import X_OK
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn import model_selection
import pickle
from sklearn.linear_model import LogisticRegression
data = pd.read_csv('diabetes.csv')
# print(data)

X = data.iloc[:,:8]
y = data.iloc[:,8]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=101)

model = LogisticRegression()
model.fit(X_train, y_train)


result = model.score(X_test, y_test)
print(result)

#save the model

pickle.dump(model, open('diab_79.pkl','wb'))
