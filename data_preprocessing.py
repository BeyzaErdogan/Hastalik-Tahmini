import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

training = pd.read_csv("Training.csv")
testing = pd.read_csv("Testing.csv")

#Null Data Check
missing_values_count = training.isnull().sum()

total_cells = np.product(training.shape)#satır*sutun(toplam hücreyi verir.)

total_missing = missing_values_count.sum()

le = preprocessing.LabelEncoder()  #3 adet kategorik değer var ise bunları 0,1,2 olarak dönüştürür.

x = training.iloc[:,:132]
y = training['prognosis']

testX = testing.iloc[:,:132]
testY = testing['prognosis']
testY = le.fit_transform(testY)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

#Decision Tree Model
clf  = DecisionTreeClassifier()
clf = clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
scores = cross_val_score(clf, x_test, y_test, cv=5)
print("for Decision Tree: ")
print (scores.mean())

#SVC Model
model = SVC()
model.fit(x_train,y_train)
print("for SVC: ")
print(model.score(x_test,y_test))
predict = model.predict(x_test)
print(accuracy_score(y_test, predict))


rfc = RandomForestClassifier(n_estimators=4, criterion='entropy')
rfc.fit(x_train,y_train)
scores2 = cross_val_score(clf, x_test, y_test, cv=5)
print("for random forest: ")
print (scores2.mean())
y_pred = rfc.predict(x_test)
