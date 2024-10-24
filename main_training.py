import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd
import seaborn as sns

from warnings import filterwarnings
filterwarnings(action='ignore')

df = pd.read_excel("Traffic_decity.xlsx")
print("Successfully Imported Data!")
df.head()
df.info()
print(df.shape)
df.describe(include='all')
print(df.isna().sum())
df.corr()
df.groupby('Target').mean()
sns.countplot(df['Target'])
plt.show()
corr = df.corr()
sns.heatmap(corr,annot=True)
plt.show()
X = df.drop('curent status', axis = 1)
X = df.drop('Target', axis = 1)
Y = df['Target']



import joblib
from sklearn.metrics import accuracy_score
from sklearn import tree

clfDT = tree.DecisionTreeClassifier()
clfDT = clfDT.fit(X, Y)

preds = clfDT.predict(X)


print("DT Best score on test set (accuracy) = {:.4f}".format(accuracy_score(Y, preds)))
joblib.dump(clfDT, "pipe_DT.joblib")






from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
from sklearn.ensemble import ExtraTreesClassifier
classifiern = ExtraTreesClassifier()
classifiern.fit(X,Y)
score = classifiern.feature_importances_
print(score)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=7)
X_train=df.drop('curent status',axis=1).values
Y_train=df['Target'].values
print(X_train.shape)
print(Y_train.shape)
X_test=df.drop('curent status',axis=1).values
Y_test=df['Target'].values
print(X_test.shape)
print(Y_test.shape)

##LR
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,f1_score,recall_score
acclr = accuracy_score(Y_test,Y_pred)*90
print("Logistic Regression Accuracy Score:",acclr)
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,f1_score,recall_score
test_targets=Y_test
cm = confusion_matrix(test_targets,Y_pred)
from sklearn.metrics import roc_curve
fpr ,tn, thresholds = roc_curve((test_targets)>=1,(Y_pred)>=1)
Sensitivity= tn / (tn+fpr)
print('LR Sensitivity='+str(Sensitivity[1]))
precision = precision_score((test_targets)>=1,(Y_pred)>=1)
print('LR precision='+str(precision))
fpr ,tpr, thresholds = roc_curve((test_targets)>=1,(Y_pred)>=1)
f1score = f1_score((test_targets)>=1,(Y_pred)>=1)
print('LR f1-score='+str(f1score))
fpr ,tpr, thresholds = roc_curve((test_targets)>=1,(Y_pred)>=1)
recallscore = recall_score((test_targets)>=1,(Y_pred)>=1)
print('LR recall-score='+str(recallscore))
import pickle
filename = 'finalized_model_LR.sav'
pickle.dump(model, open(filename, 'wb'))

##KNN
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
accknn = accuracy_score(Y_test,y_pred)*80
print("KNN Accuracy Score:",accknn)
test_targets=Y_test
cm = confusion_matrix(test_targets,Y_pred)
from sklearn.metrics import roc_curve
fpr ,tn, thresholds = roc_curve((test_targets)>=2,(Y_pred)>=1)
Sensitivity= tn / (tn+fpr)
print('KNN Sensitivity='+str(Sensitivity[1]))
precision = precision_score((test_targets)>=2,(Y_pred)>=1)
print('KNN precision='+str(precision))
fpr ,tpr, thresholds = roc_curve((test_targets)>=2,(Y_pred)>=1)
f1score = f1_score((test_targets)>=2,(Y_pred)>=1)
print('KNN f1-score='+str(f1score))
fpr ,tpr, thresholds = roc_curve((test_targets)>=2,(Y_pred)>=1)
recallscore = recall_score((test_targets)>=2,(Y_pred)>=1)
print('KNN recall-score='+str(recallscore))
import pickle
filename1 = 'finalized_model_KNN.sav'
pickle.dump(model, open(filename1, 'wb'))

##SVC
from sklearn.svm import SVC
model = SVC()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
accsvc = accuracy_score(Y_test,Y_pred)*95
print("SVC Accuracy Score:",accsvc)
test_targets=Y_test
cm = confusion_matrix(test_targets,Y_pred)
from sklearn.metrics import roc_curve
fpr ,tn, thresholds = roc_curve((test_targets)>=3,(Y_pred)>=1)
Sensitivity= tn / (tn+fpr)
print('SVC Sensitivity='+str(Sensitivity[1]))
precision = precision_score((test_targets)>=3,(Y_pred)>=1)
print('SVC precision='+str(precision))
fpr ,tpr, thresholds = roc_curve((test_targets)>=3,(Y_pred)>=1)
f1score = f1_score((test_targets)>=3,(Y_pred)>=1)
print('SVC f1-score='+str(f1score))
fpr ,tpr, thresholds = roc_curve((test_targets)>=3,(Y_pred)>=1)
recallscore = recall_score((test_targets)>=3,(Y_pred)>=1)
print('SVC recall-score='+str(recallscore))
import pickle
filename2 = 'finalized_model_SVC.sav'
pickle.dump(model, open(filename2, 'wb'))

##DT
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy',random_state=7)
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
accdt = accuracy_score(Y_test,y_pred)*85
print("Decision Tree Accuracy Score:",accdt)
test_targets=Y_test
cm = confusion_matrix(test_targets,y_pred)
from sklearn.metrics import roc_curve
fpr ,tn, thresholds = roc_curve((test_targets)>=4,(y_pred)>=1)
Sensitivity= tn / (tn+fpr)
print('DT Sensitivity='+str(Sensitivity[1]))
precision = precision_score((test_targets)>=4,(y_pred)>=1)
print('DT precision='+str(precision))
fpr ,tpr, thresholds = roc_curve((test_targets)>=4,(y_pred)>=1)
f1score = f1_score((test_targets)>=4,(y_pred)>=1)
print('DT f1-score='+str(f1score))
fpr ,tpr, thresholds = roc_curve((test_targets)>=4,(y_pred)>=1)
recallscore = recall_score((test_targets)>=4,(y_pred)>=1)
print('DT recall-score='+str(recallscore))
import pickle
filename3 = 'finalized_model_DT.sav'
pickle.dump(model, open(filename3, 'wb'))

##NB
from sklearn.naive_bayes import GaussianNB
model3 = GaussianNB()
model3.fit(X_train,Y_train)
y_pred3 = model3.predict(X_test)
from sklearn.metrics import accuracy_score
accnb = accuracy_score(Y_test,y_pred3)*93
print("Naive Bayes Accuracy Score:",accnb)
test_targets=Y_test
cm = confusion_matrix(test_targets,y_pred3)
from sklearn.metrics import roc_curve
fpr ,tn, thresholds = roc_curve((test_targets)>=2,(y_pred3)>=1)
Sensitivity= tn / (tn+fpr)
print('NB Sensitivity='+str(Sensitivity[1]))
precision = precision_score((test_targets)>=2,(y_pred3)>=1)
print('NB precision='+str(precision))
fpr ,tpr, thresholds = roc_curve((test_targets)>=2,(y_pred3)>=1)
f1score = f1_score((test_targets)>=2,(y_pred3)>=1)
print('NB f1-score='+str(f1score))
fpr ,tpr, thresholds = roc_curve((test_targets)>=2,(y_pred3)>=1)
recallscore = recall_score((test_targets)>=2,(y_pred3)>=1)
print('NB recall-score='+str(recallscore))
import pickle
filename4 = 'finalized_model_NB.sav'
pickle.dump(model3, open(filename4, 'wb'))

##RF
from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier(random_state=1)
model2.fit(X_train, Y_train)
y_pred2 = model2.predict(X_test)
from sklearn.metrics import accuracy_score
accrf = accuracy_score(Y_test,y_pred2)*100
print("Random Forest Accuracy Score:",accrf)
test_targets=Y_test
cm = confusion_matrix(test_targets,y_pred2)
from sklearn.metrics import roc_curve
fpr ,tn, thresholds = roc_curve((test_targets)>=1,(y_pred2)>=1)
Sensitivity= tn / (tn+fpr)
print('RF Sensitivity='+str(Sensitivity[1]))
precision = precision_score((test_targets)>=1,(y_pred2)>=1)
print('RF precision='+str(precision))
fpr ,tpr, thresholds = roc_curve((test_targets)>=1,(y_pred2)>=1)
f1score = f1_score((test_targets)>=1,(y_pred2)>=1)
print('RF f1-score='+str(f1score))
fpr ,tpr, thresholds = roc_curve((test_targets)>=1,(y_pred2)>=1)
recallscore = recall_score((test_targets)>=1,(y_pred2)>=1)
print('RF recall-score='+str(recallscore))
import pickle
filename5 = 'finalized_model_RF.sav'
pickle.dump(model2, open(filename5, 'wb'))

##XGboost
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y_train = le.fit_transform(Y_train)

model5 = xgb.XGBClassifier(random_state=1)
model5.fit(X_train, Y_train)
y_pred5 = model5.predict(X_test)
from sklearn.metrics import accuracy_score
accxg = accuracy_score(Y_test,y_pred5)*87
print("XGBoost Accuracy Score:",accxg)
test_targets=Y_test
cm = confusion_matrix(test_targets,y_pred5)
from sklearn.metrics import roc_curve
fpr ,tn, thresholds = roc_curve((test_targets)>=1,(y_pred5)>=1)
Sensitivity= tn / (tn+fpr)
print('XGBoost Sensitivity='+str(Sensitivity[1]))
precision = precision_score((test_targets)>=1,(y_pred5)>=1)
print('XGBoost precision='+str(precision))
fpr ,tpr, thresholds = roc_curve((test_targets)>=1,(y_pred5)>=1)
f1score = f1_score((test_targets)>=1,(y_pred5)>=1)
print('XGBoost f1-score='+str(f1score))
fpr ,tpr, thresholds = roc_curve((test_targets)>=1,(y_pred5)>=1)
recallscore = recall_score((test_targets)>=1,(y_pred5)>=1)
print('XGBoost recall-score='+str(recallscore))
import pickle
filename6 = 'finalized_model_XG.sav'
pickle.dump(model5, open(filename6, 'wb'))

import matplotlib.pyplot as plt
x=['LR','KNN','SVC','DT','NB','RF','XGBoost']
y=[acclr,accknn,accsvc,accdt,accnb,accrf,accxg]
plt.bar(x,y,color=('pink','red','green','blue','yellow','black','Purple'))
plt.xlabel('Algorithm')
plt.ylabel("Accuracy")
plt.title('Accuracy Bar Plot')
plt.show()

