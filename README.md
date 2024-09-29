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
Developed by:MUKITHA V M 
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
![369268104-9d88a883-db85-4036-b3f8-3f27ce8dab33](https://github.com/user-attachments/assets/a52dd8fd-0ae9-405d-97ab-a638e1e4d276)
## ACCURACY :
![369268140-cf464349-aa04-443d-a466-26e214d95951](https://github.com/user-attachments/assets/fea9f6f0-4cc9-4875-a4a9-3b78d925ee60)
## New predicted:
![369268175-ce36cfee-0ddf-4d82-be97-ad7028eccfb5](https://github.com/user-attachments/assets/b1c08f94-59fe-4f4c-9ecb-ba127e799fb6)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
