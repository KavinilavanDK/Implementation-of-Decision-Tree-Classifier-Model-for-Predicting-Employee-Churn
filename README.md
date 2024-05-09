# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import pandas module and import the required data set.
2.Find the null values and count them.
3.Count number of left values.
4.From sklearn import LabelEncoder to convert string values to numerical values. 5.From sklearn.model_selection import train_test_split. 6.Assign the train dataset and test dataset. 7.From sklearn.tree import DecisionTreeClassifier. 8.Use criteria as entropy. 9.From sklearn import metrics. 10.Find the accuracy of our model and predict the require values.


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: KAVI NILAVAN DK
RegisterNumber:  21222330103
*/
```
```
``
import pandas as pd
data=pd.read_csv("/content/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```
## Output:
#### Data.head()
![11](https://github.com/KavinilavanDK/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144870429/f8854bff-ef08-4cc7-a66a-8a16e11dc647)

#### Data.info():
![22](https://github.com/KavinilavanDK/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144870429/634237ae-3fcd-4b3c-9c37-4f7b4afa2303)

#### isnull() and sum():
![33](https://github.com/KavinilavanDK/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144870429/92213f0b-b854-4c29-ab79-8987b9cd5b67)

#### Data Value Counts():
![44](https://github.com/KavinilavanDK/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144870429/c7113d09-66b3-4db4-822a-7189a8737426)

#### Data.head() for salary:
![55](https://github.com/KavinilavanDK/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144870429/fece630a-ecd3-48f7-831f-f73978ca1242)

#### x.head():
![66](https://github.com/KavinilavanDK/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144870429/20f5d322-d3c6-411e-aacc-8db027db21c1)

#### Accuracy Value:
![88](https://github.com/KavinilavanDK/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144870429/3204e053-024c-4c9b-b2bb-18f1bfd9385c)

#### Data Prediction:
![99](https://github.com/KavinilavanDK/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144870429/549780e5-0e32-4d72-8984-da56a28cc060)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
