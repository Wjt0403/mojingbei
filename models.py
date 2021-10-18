import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV

import xgboost as xgb
from sklearn import metrics
import ipdb
from sklearn.preprocessing import StandardScaler
from xgboost.sklearn import XGBClassifier
from mlxtend.regressor import StackingRegressor
from mlxtend.data import boston_housing_data
import pickle
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def standardX(x):
    return StandardScaler().fit_transform(x)

def make_pred(alg,X_test,y_test,save_path):
    y_predprob = alg.predict_proba(X_test)[:,1]
    myplot_auc(y_test, y_predprob,save_path)

def myplot_auc(y_test, y_pred_proba,save_path):
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_proba)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(6,6))
    plt.title('Validation ROC')
    plt.plot(fpr, tpr, 'b', label = 'Val AUC = %0.3f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(save_path)
    plt.show()

def model_xgb():
    data = pd.read_csv('data_xgb.csv')
    x = data.drop('target',1)
    y = data['target']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    x_train = standardX(x_train)
    x_test = standardX(x_test)
    Xgb = XGBClassifier(learning_rate=0.01,n_estimators=1000,max_depth=3,min_child_weight=5,gamma=0,subsample=0.9,colsample_bytree=0.7,objective= 'binary:logistic',scale_pos_weight=10,seed=27,reg_alpha=0.01,eval_metric=['logloss','auc','error'])  
    Xgb.fit(x_train,y_train)
    pickle.dump(Xgb, open("./models/xgb/xgb.dat", "wb"))
    make_pred(Xgb,x_test,y_test,'./data/models/xgb/xgb_auc.jpg')

def grid_search(x_train,y_train,model,param_test):
    gsearch = GridSearchCV(estimator=model, param_grid=param_test, scoring='roc_auc', verbose=100,cv=5)  
    gsearch.fit(x_train,y_train)  
    gsearch.best_params_, gsearch.best_score_  

if __name__ == "__main__":
    #ipdb.set_trace()
    data = pd.read_csv('data_xgb.csv')
    