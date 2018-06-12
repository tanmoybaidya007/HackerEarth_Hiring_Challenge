print("Libraries Importing... \n")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
#import tensorflow
#import seaborn as sns
#import xgboost
#import catboost
#import lightgbm
#import keras
#%matplotlib inline

print("Datasets Importing... \n")

train=pd.read_csv("Data/train.csv")
test=pd.read_csv("Data/test.csv")

train=train[['project_id','goal','disable_communication','country','currency','deadline','state_changed_at','created_at','launched_at','backers_count','final_status']]
test=test[['project_id','goal','disable_communication','country','currency','deadline','state_changed_at','created_at','launched_at']]

# Data Preprocessing and Feature Engineering

 # Datetime Object

train.deadline=pd.to_datetime(train.deadline,unit='s')
test.deadline=pd.to_datetime(test.deadline,unit='s')

train.state_changed_at=pd.to_datetime(train.state_changed_at,unit='s')
test.state_changed_at=pd.to_datetime(test.state_changed_at,unit='s')

train.created_at=pd.to_datetime(train.created_at,unit='s')
test.created_at=pd.to_datetime(test.created_at,unit='s')

train.launched_at=pd.to_datetime(train.launched_at,unit='s')
test.launched_at=pd.to_datetime(test.launched_at,unit='s')

train['launch_diff']=(train.launched_at.dt.year-train.created_at.dt.year)*365+(train.launched_at.dt.month-train.created_at.dt.month)*30+(train.launched_at.dt.day-train.created_at.dt.day)
test['launch_diff']=(test.launched_at.dt.year-test.created_at.dt.year)*365+(test.launched_at.dt.month-test.created_at.dt.month)*30+(test.launched_at.dt.day-test.created_at.dt.day)

train['success_diff']=(train.state_changed_at.dt.year-train.created_at.dt.year)*365+(train.state_changed_at.dt.month-train.created_at.dt.month)*30+(train.state_changed_at.dt.day-train.created_at.dt.day)
test['success_diff']=(test.state_changed_at.dt.year-test.created_at.dt.year)*365+(test.state_changed_at.dt.month-test.created_at.dt.month)*30+(test.state_changed_at.dt.day-test.created_at.dt.day)

train['deadline_diff']=(train.deadline.dt.year-train.created_at.dt.year)*365+(train.deadline.dt.month-train.created_at.dt.month)*30+(train.deadline.dt.day-train.created_at.dt.day)
test['deadline_diff']=(test.deadline.dt.year-test.created_at.dt.year)*365+(test.deadline.dt.month-test.created_at.dt.month)*30+(test.deadline.dt.day-test.created_at.dt.day)

train['launch_success_diff']=(train.state_changed_at.dt.year-train.launched_at.dt.year)*365+(train.state_changed_at.dt.month-train.launched_at.dt.month)*30+(train.state_changed_at.dt.day-train.launched_at.dt.day)
test['launch_success_diff']=(test.state_changed_at.dt.year-test.launched_at.dt.year)*365+(test.state_changed_at.dt.month-test.launched_at.dt.month)*30+(test.state_changed_at.dt.day-test.launched_at.dt.day)

train['launch_deadline_diff']=(train.deadline.dt.year-train.launched_at.dt.year)*365+(train.deadline.dt.month-train.launched_at.dt.month)*30+(train.deadline.dt.day-train.launched_at.dt.day)
test['launch_deadline_diff']=(test.deadline.dt.year-test.launched_at.dt.year)*365+(test.deadline.dt.month-test.launched_at.dt.month)*30+(test.deadline.dt.day-test.launched_at.dt.day)

train=train[['project_id', 'goal', 'disable_communication', 'country', 'currency',
       'launch_diff', 'success_diff','deadline_diff', 'launch_deadline_diff', 'launch_success_diff','backers_count', 'final_status',]]

test=test[['project_id', 'goal', 'disable_communication', 'country', 'currency',
       'launch_diff', 'success_diff','deadline_diff', 'launch_deadline_diff', 'launch_success_diff']]

## Currency Converter

def currency_converter(currency):
    x=currency[0]
    y=currency[1]
    if x=='AUD' or x=='CAD':
        return(y*0.80)
    elif x=='CHF':
        return(y*1.04)
    elif x=='DKK':
        return(y*0.16)
    elif x=='EUR':
        return(y*1.22)
    elif x=='GBP':
        return(y*1.40)
    elif x=='HKD':
        return(y*0.13)
    elif x=='MXN':
        return(y*0.05)
    elif x=='NOK':
        return(y*0.13)
    elif x=='NZD':
        return(y*0.73)
    elif x=='SEK':
        return(y*0.12)
    elif x=='SGD':
        return(y*0.76)
    else:
        return(y)

train['goal']=train[['currency','goal']].apply(lambda x: currency_converter(x),axis=1)
test['goal']=test[['currency','goal']].apply(lambda x: currency_converter(x),axis=1)

train.drop(['currency'],axis=1,inplace=True)
test.drop(['currency'],axis=1,inplace=True)

 ## Country Column
def country(x):
    if x=='US':
        return(1)
    else:
        return(0)

train.country=train.country.apply(lambda x: country(x))
test.country=test.country.apply(lambda x: country(x))

## Disable Communication Column

dc={True:0,False:1}
train.disable_communication=train.disable_communication.apply(lambda x: dc[x])
test.disable_communication=test.disable_communication.apply(lambda x: dc[x])

train.to_csv("train_20_Jan.csv",index=False)
test.to_csv("test_20_Jan.csv",index=False)

X=train.drop(['project_id','backers_count','final_status'],axis=1)
y=train.final_status

test_data=(test.iloc[:,1:])

from sklearn.model_selection import cross_val_score,train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)


from sklearn.ensemble import RandomForestClassifier
model_RF=RandomForestClassifier()
model_RF.fit(X_train,y_train)

### Scoring Function
def Score(model,X_train,y_train,X_test,y_test,train=True):
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score
    scoring = ['accuracy','precision', 'recall','f1']
    if train==True:
        print("Training Result \n")
        print("Accuracy Score:  {0:0.4f} \n".format(accuracy_score(y_train,model.predict(X_train))))
        scores=cross_val_score(estimator=model,X=X_train,y=y_train,cv=15,scoring='accuracy',n_jobs=40)
        print("Cross-Validation Score: \n",scores.mean())
        print("Standard Deviation: \n",scores.std())
    elif train==False:
        print("TestResult \n")
        print("Accuracy Score:  {0:0.4f} \n".format(accuracy_score(y_test,model.predict(X_test))))

Score(model_RF,X_train,y_train,X_test,y_test,train=True)

Score(model_RF,X_train,y_train,X_test,y_test,train=False)

print("Grid Search Started.....\n")

##Grid Search
from sklearn.grid_search import GridSearchCV

param_grid={
            "n_estimators":[100,150,200,300,500],
            "max_features":["auto",'log2',0.1,0.2,0.3,0.5],
            "min_samples_leaf":[10,20,30,50]
           }
        
grid_search=GridSearchCV(estimator=model_RF,param_grid=param_grid,cv=10,n_jobs=40)
grid_search.fit(X_train,y_train)

print("Grid Search Done.....\n")

print("Grid Search Best Score: \n",grid_search.best_score_)
parametrs=grid_search.best_params_
print("Grid Search Best Parametrs: \n",parametrs)


print("Final Modelling Started.....\n")
## Final RandomForest Model

model_RF=grid_search.best_estimator_
model_RF.fit(X_train,y_train)

Score(model_RF,X_train,y_train,X_test,y_test,train=True)
Score(model_RF,X_train,y_train,X_test,y_test,train=False)

pred_RF=model_RF.predict(test_data)

print("Submission Process Started.....\n")

sub=pd.read_csv("Data/samplesubmission.csv")
sub.final_status=Pred_RF
sub.to_csv("RF_Base.csv",index=False)

print("All Done.....\n")

















