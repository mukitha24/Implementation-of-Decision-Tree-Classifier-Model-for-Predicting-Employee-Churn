# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program
2.attach the given data file
3.now find the satisfaction level of employee data
4.find the accuracy and new predict value 5.end the program
 
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: MUKITHA V M
RegisterNumber: 212223040119 
*/
```
```
import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/Employee.csv")
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
![369268104-9d88a883-db85-4036-b3f8-3f27ce8dab33](https://github.com/user-attachments/assets/e2455ca1-36f7-440b-afd1-21b10444ba4a)
## ACCURACY:
![369268140-cf464349-aa04-443d-a466-26e214d95951](https://github.com/user-attachments/assets/76c48d25-1011-4d59-8d7c-361cac148f14)
## New predicted:
![369268175-ce36cfee-0ddf-4d82-be97-ad7028eccfb5](https://github.com/user-attachments/assets/de2fc9f2-6cad-4658-8c57-57fe8e20da94)
## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
