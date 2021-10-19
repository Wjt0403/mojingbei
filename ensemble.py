from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_classification
from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt
from sklearn import metrics
from models import myplot_auc, make_pred,load_data
from sklearn.metrics import roc_curve,auc
from sklearn import datasets 
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import pickle
import ipdb

def load_model(path):
    loaded_model = pickle.load(open(path, "rb"))
    return loaded_model    
def normalize(v):
    return v/sum(v)


def  stacking_ensemble():
    x_train, x_test, y_train, y_test = load_data(False,False,True)
    clf1 = KNeighborsClassifier(n_neighbors=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()
    lr = LogisticRegression()
    sclf = StackingClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=lr)
    sclf.fit(x_train,y_train)
    pickle.dump(sclf, open("./models/stacking/stacking.dat", "wb"))
    make_pred(sclf,x_test,y_test,'./models/stacking/stacking_auc.jpg')

def auc_test():
    data, target = make_blobs(n_samples=20, centers=2, random_state=1, cluster_std=1.0 )
    X_train,X_test,y_train,y_test = train_test_split(data, target, test_size=0.2, random_state=1)
    lr = LogisticRegression()
    lr.fit(X_train,y_train)
    y_pred = lr.predict(X_test)
    y_pred_proba =lr.predict_proba(X_test)
    print('y_pred = ',y_pred)
    print('y_pred_proba = ',y_pred_proba)
    print(y_pred_proba.shape)
    fpr1, tpr1, thresholds1 = roc_curve(y_test, y_pred, pos_label=1)
    fpr2, tpr2, thresholds2 = roc_curve(y_test, y_pred_proba[:,1], pos_label=1)
    print("y_pred auc:",auc(fpr1, tpr1))
    print("y_pred_proba auc:",auc(fpr2, tpr2))

def blending_voting(paths_list_models,voting_coef):
    x_train, x_test, y_train, y_test = load_data(False,False,True)
    models = []
    for path in paths_list_models:
        model = load_model(path)
        models.append(model)
    pred_proba_list = []
    for model in models:
        pred_proba = model.predict_proba(x_test)[:,1]
        pred_proba_list.append(pred_proba)
    voting_proba = np.zeros((len(pred_proba_list[0]),))
    for i in range(len(models)):
        voting_proba += voting_coef[i]*pred_proba_list[i]
    fpr, tpr, thresholds = roc_curve(y_test, voting_proba, pos_label=1)
    print("voting auc:",auc(fpr, tpr))
    


if __name__ == "__main__":
    
    paths_list_models = ["./models/lgb/lgb.dat","./models/xgb/xgb.dat","./models/gbc/gbc.dat"]
    voting_coef = normalize(np.array([0.753,0.750,0.752]))
    blending_voting(paths_list_models,voting_coef)
    
    #ipdb.set_trace()
    '''
    x_train, x_test, y_train, y_test = load_data(False,False,True)
    lgb = load_model("./models/lgb/lgb.dat")
    xgb = load_model("./models/xgb/xgb.dat")
    lgb_pred_proba = lgb.predict_proba(x_test)[:,1]
    xgb_pred_proba = xgb.predict_proba(x_test)[:,1]
    ini = np.zeros((len(lgb_pred_proba),))
    print(ini+lgb_pred_proba*2+xgb_pred_proba)    
    '''



