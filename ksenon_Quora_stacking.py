#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 18:19:05 2017

@author: pavel
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from collections import defaultdict
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold,KFold
from sklearn import linear_model
from sklearn.metrics import log_loss
from scipy.sparse import csr_matrix
import lightgbm as lgb 
# reload(sys)
# sys.setdefaultencoding('utf8')
import string
import random
import math
###
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier 

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

import nltk
from nltk.corpus import stopwords
from collections import Counter
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import xgboost as xgb 
from sklearn.preprocessing import Imputer 


# Get our train target

train_data = pd.read_csv('/home/pavel/anaconda3/Scripts/Quora/train.csv') 
Y_train = train_data['is_duplicate'].values

test_data = pd.read_csv('/home/pavel/anaconda3/Scripts/Quora/test.csv')

### Load train-test datasets ###

X_train_pred = pd.read_csv('/home/pavel/anaconda3/Scripts/Quora/X_train_predict_new.csv')  
X_test_pred = pd.read_csv('/home/pavel/anaconda3/Scripts/Quora/X_test_predict_new.csv')  


X_train_fin = pd.read_csv('/home/pavel/anaconda3/Scripts/Quora/X_train_fin.csv')  
X_test_fin = pd.read_csv('/home/pavel/anaconda3/Scripts/Quora/X_test_fin.csv')  


# X_train_fin = X_train_fin.fillna(0) 
# X_test_fin = X_test_fin.fillna(0) 

### PageRank FEATURE ###

# train_features = pd.read_csv('/home/pavel/anaconda3/Scripts/Quora/pagerank_train.csv') 
# test_features = pd.read_csv('/home/pavel/anaconda3/Scripts/Quora/pagerank_test.csv') 



# Save final feature datasets
 X_train_pred.to_csv('X_train_light.csv', index=None) 
X_test_pred.to_csv('X_test_light.csv', index=None)   
# New light datasets


####### Calibrate our target #########

 pos_train = X_train_preds_new[Y_train == 1]
 neg_train = X_train_preds_new[Y_train == 0]

# Now we oversample the negative class
# There is likely a much more elegant way to do this...
 p = 0.165
 scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
 while scale > 1:
    neg_train = pd.concat([neg_train, neg_train])
    scale -=1
 neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
 print(len(pos_train) / (len(pos_train) + len(neg_train)))

 X_train_fix = pd.concat([pos_train, neg_train])
 Y_train_fix = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
 del pos_train, neg_train


# Y_train_new = pd.DataFrame()
# Y_train_new['is_duplicate'] = Y_train_fix


X_train_pred.columns 

# Finally, we split some of the data off for validation
from sklearn.cross_validation import train_test_split 

 x_train, x_valid, y_train, y_valid = train_test_split(X_train_fix, Y_train_fix, test_size=0.2, random_state=12) 

# Set our parameters for xgboost
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02  # 0.015 # 0.02
params['max_depth'] = 7
params['subsample'] = 0.7
# params['base_score'] = 0.2 
#params['colsample_bylevel'] = 0.7
# params['max_delta_step'] = 0.36
#params['scale_pos_weight'] = 0.360

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(params, d_train, 2500, watchlist, early_stopping_rounds=50, verbose_eval=10) 

# 1. Val_score = 0.1694; Train Score = 0.14.. LB = 0.1556
# 2. Val_score = 0.1678;  LB = 0.1555

# Add PageRank
# 3. Val_score = 0.1687; Train Score = 0.1369; LB = 0.1546
# 4. Val_score = 0.1662; Train score = 0.1378; LB = 0.1489 (2*xgb+lstm)
# 5. Val_score = 0.1656; Train score = 0.1381; LB = 0.

# Stacking by my hands (1 level - LGBM, MLP, RF; 2 level - XGB)
# Val_score = 0.1566; Train_score = 0.1279 LB = 0.1610 eta:0.02
# Val score = 0.1655; Train_score = 0.1570 LB = 0.1573 eta:0.04
# Val score = 0.1639; Train_score = 0.1516; LB= 0.15903

# Val score = 0.1689; Train score = 0.1397; LB = 

## Make predictions ##

d_test = xgb.DMatrix(X_test_preds_new) 
p_test = bst.predict(d_test)  

sub = pd.DataFrame()
sub['test_id'] = test_data['test_id']
sub['is_duplicate'] = p_test

sub.to_csv('quora_xgb_stacking_v5.0.csv', index=False)  





############################################################
 


X_train_fix = Imputer().fit_transform(X_train_fix)
X_train_new = pd.DataFrame(X_train_fix) 





from sklearn.model_selection import KFold, StratifiedKFold 
### HyperOpt ###
Y_train_new = pd.DataFrame()

Y_train_new['is_duplicate'] = Y_train_fix



def objective_xgb(space):
    
    numfolds = 5
    total = 0
    kf = StratifiedKFold(n_splits=numfolds, shuffle=True,random_state=666)
            
    
    clf = xgb.XGBClassifier(n_estimators = 100, 
                            max_depth = space['max_depth'],
                            learning_rate = space['learning_rate'],
                            min_child_weight = space['min_child_weight'],
                            subsample = space['subsample'],
                            colsample_bytree = space['colsample_bytree'])
#                            colsample_bylevel = space['colsample_bylevel'],
#                           nthread = -1)

    
    for train_index, test_index in kf.split(X_train_pred,Y_train_new.is_duplicate):
        xtrain, xtest = X_train_pred.iloc[train_index], X_train_pred.iloc[test_index]
        ytrain, ytest = Y_train_new.iloc[train_index], Y_train_new.iloc[test_index]
        
        eval_set = [(xtrain, ytrain),(xtest, ytest)]

        clf.fit(xtrain, ytrain.values.ravel(), eval_metric="logloss",eval_set = eval_set, early_stopping_rounds=50)
 
#        rf.fit(xtrain,ytrain.values.ravel())
 
        pred = clf.predict_proba(xtest)[:,1]
     
        logloss = log_loss(ytest, pred)
        print ("SCORE:", logloss)  
        total += logloss
    total = total/numfolds
    print (total)
    return{'loss':total, 'status': STATUS_OK }




space ={
    'max_depth': hp.choice('max_depth', range(6,9)),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'min_child_weight': hp.quniform ('min_child_weight', 1, 10, 1),
    'subsample': hp.uniform ('x_subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.5, 1)
    }

  
    
trials = Trials()  

best = fmin(fn=objective_xgb,
            space=space,
            algo=tpe.suggest,
            max_evals=20, 
            trials=trials) 


print (best)   

trials.trials
trials.best_trial 


#new
# 0.1649 {'learning_rate': 0.16798864789130424,
# 'colsample_by_tree': 0.9699281358559593, 'min_child_weight': 7.0, 'max_depth': 2, 'x_subsample': 0.919491248708016}



# {'learning_rate': 0.1866301239886971, 'max_depth': 2, 'min_child_weight': 6.0,
# 'colsample_by_tree': 0.8279436758422373, 'x_subsample': 0.9707456665175549}

# n_est = 100; Val Score = 0.1695; LB = 0.15905
# n_est = 1000; Train Score = 0.1696 Val_score = 0.1788 LB = 0.1574
# n_est = 2000; Train Score = 0.1693 Val score = 0.1780LB =0.1573






### XGB Classifier ###

    numfolds = 5
    total = 0 
    kf = StratifiedKFold(n_splits=numfolds, shuffle=True,random_state=4321) 
    
    xgb_model = xgb.XGBClassifier(n_estimators = 300, 
                            max_depth = 5,  # 7
                            learning_rate = 0.05, # 0.168
                            min_child_weight = 7,
                            subsample = 0.97,
                            colsample_bytree = 0.82794) 

 

del X_train_pred['pred_lgbm']
del X_test_pred['pred_lgbm']
   

pred_test_full = np.zeros(X_test_pred.shape[0]) 

    for train_index, test_index in kf.split(X_train_preds_new,Y_train_new.is_duplicate):
        xtrain, xtest = X_train_preds_new.iloc[train_index], X_train_preds_new.iloc[test_index]
        ytrain, ytest = Y_train_new.iloc[train_index], Y_train_new.iloc[test_index]
        
        eval_set = [(xtrain, ytrain),(xtest, ytest)]


        xgb_model.fit(xtrain, ytrain.values.ravel(), eval_metric="logloss",eval_set = eval_set, early_stopping_rounds=50)
#         rf.fit(xtrain, ytrain)
        
        pred_test = xgb_model.predict_proba(X_test_pred)[:,1]
        pred_test_full += pred_test
        


        pred = xgb_model.predict_proba(xtest)[:,1]
        
        logloss = log_loss(ytest, pred)
#        print ("SCORE:", logloss)  
        total += logloss
    total = total/numfolds
    print (total)

pred_test_full /= 5 


FeatImportance = pd.DataFrame({'feature':X_train_pred.columns,'importance':np.round(bst.feature_importances_,3)})
FeatImportance = FeatImportance.sort_values('importance',ascending=False).set_index('feature')
print (FeatImportance)
FeatImportance.plot.bar()



# train - 0.1571 val - 0.1645
# train - 0.1640 val - 0.1664 n_est = 100



# train - 0.1643 val - 0.1665 LB - 0.1578 without neural nets
# train - 0.1649; val - 0.1653; LB - 

## Predictions ###


 Pred_prob = xgb_model.predict_proba(X_test_pred)[:,1]  


p_test2 = pd.DataFrame(pred_test_full)  


sub = pd.DataFrame()

sub['test_id'] = test_data['test_id']
sub['is_duplicate'] = p_test2

sub.to_csv('quora_xgb_stacking_v4.3.csv',index=None) 

































def objective(space):
    
    numfolds = 5
    total = 0
    kf = StratifiedKFold(n_splits=numfolds, shuffle=True,random_state=13)
            
    
    rf = RandomForestClassifier(n_estimators = 200, 
                            max_depth = space['max_depth'],
                            max_features = space['max_features'],
                            criterion = space['criterion'],
                            min_impurity_split = 0.0005,
#                            min_impurity_split = space['min_impurity_split'],
             #               scale = space['scale'],
             #               normalize = space['normalize'],
             #               min_samples_leaf = space['min_samples_leaf'],
             #               min_weight_fraction_leaf  = space['min_weight_fraction_leaf'],
             #               min_impurity_split = space['min_impurity_split'],
                            random_state = 666,
             #               warm_start = True,                            
                            n_jobs = -1
                            )
    
    
    for train_index, test_index in kf.split(X_train_new,Y_train_new.is_duplicate):
        xtrain, xtest = X_train_new.iloc[train_index], X_train_new.iloc[test_index]
        ytrain, ytest = Y_train_new.iloc[train_index], Y_train_new.iloc[test_index]
        
#        eval_set = [(xtrain, ytrain),(xtest, ytest)]

#        clf.fit(xtrain, ytrain, eval_metric="logloss",eval_set = eval_set, early_stopping_rounds=50)
 
        rf.fit(xtrain,ytrain.values.ravel())
 
        pred = rf.predict_proba(xtest)[:,1]
     
        logloss = log_loss(ytest, pred)
        print ("SCORE:", logloss)  
        total += logloss
    total = total/numfolds
    print (total)
    return{'loss':total, 'status': STATUS_OK }


space ={
    'max_depth': hp.choice('max_depth', range(1,15)),
    'max_features': hp.uniform('max_features',0,0.7),
#   'n_estimators': hp.choice('n_estimators', range(1,50)),
    'criterion': hp.choice('criterion', ["gini", "entropy"]),
    }




  
    
trials = Trials()

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=10, 
            trials=trials) 

print (best)   

trials.trials
trials.best_trial




# Best parametres
# 1. {'criterion': 0, 'max_features': 0.3114671350604435, 'max_depth': 11} Val: 0.19612
# 2.






# RandomForests Classifier

from sklearn.model_selection import KFold, StratifiedKFold  

Y_train_new = pd.DataFrame()
Y_train_new['is_duplicate'] = Y_train


from sklearn.preprocessing import Imputer 

X_train_fix = Imputer().fit_transform(X_train_fin)
X_train_new = pd.DataFrame(X_train_fix)

X_test_fix =  Imputer().fit_transform(X_test_fin)
X_test_new = pd.DataFrame(X_test_fix)

    numfolds = 5
    total = 0
    kf = StratifiedKFold(n_splits=numfolds, shuffle=True,random_state=4321) # DONT TOUCH THIS !!!!!
    

ROUNDS  = 1000    
    
param = {}
param['learning_rate'] = 0.04
param['boosting_type'] = 'gbdt' # gbdt or goss
param['objective'] = 'binary'
param['metric'] = 'binary_logloss'
param['max_depth']= 7
param['sub_feature'] = 0.7 # 0.7 tr:0.1414 vl: 0.1667
# param['num_leaves'] = 1024
param['min_data'] = 100
param['min_hessian'] = 1



#lgbm = lgb.LightGBM()

# lgbm = lgb.train(param, xg_train,ROUNDS, watchlist, early_stopping_rounds=50, verbose_eval = 10) 


    rf = RandomForestClassifier(n_estimators = 200, 
                            max_depth = 11,
                            max_features = 0.3114671350604435,
                            criterion = "gini",
                            min_impurity_split = 0.0005,
                            random_state = 666                          
                            )
    
    clf = MLPClassifier(solver='adam', alpha=1e-5,activation='logistic',
                     hidden_layer_sizes=(300, 3),
                     shuffle=True,
                     random_state=4242) 
    
    
# Build for RF model
pred_train_rf = np.zeros(X_train_fin.shape[0]) 
pred_test_full = np.zeros(X_test_fin.shape[0]) 
# Build for MLP model
pred_train_clf = np.zeros(X_train_new.shape[0]) 
pred_test_full_clf = np.zeros(X_test_new.shape[0]) 
# Build for LGBM model
pred_train_lgbm = np.zeros(X_train_new.shape[0])
pred_test_full_lgbm = np.zeros(X_test_new.shape[0])     



     total = 0 
    for train_index, test_index in kf.split(X_train_new,Y_train_new.is_duplicate):
        xtrain, xtest = X_train_new.iloc[train_index], X_train_new.iloc[test_index]
        ytrain, ytest = Y_train_new.iloc[train_index], Y_train_new.iloc[test_index]
        
  #      eval_set = [(xtrain, ytrain),(xtest, ytest)]

# RandomForest Classifier
#        rf.fit(xtrain, ytrain.values.ravel()) 
  
#        pred = rf.predict_proba(xtest)[:,1]               # Predictions for out-of-fold
#        pred_train_rf[test_index] = pred                  # Assign OOF predictions to train pred dataset
#        pred_test_rf = rf.predict_proba(X_test_new)[:,1]  # Predicting on X_test
#        pred_test_full += pred_test_rf                    # Cumulate preds for all folds

# MLP Classifier
        
 #       clf.fit(xtrain, ytrain.values.ravel())
        
#        pred_clf = clf.predict_proba(xtest)[:,1]               # Predictions for out-of-fold
#        pred_train_clf[test_index] = pred_clf                 # Assign OOF predictions to train pred dataset
#        pred_test_clf = clf.predict_proba(X_test_new)[:,1]  # Predicting on X_test
#        pred_test_full_clf += pred_test_clf                    # Cumulate preds for all folds


# LightGBM Classifier

        xg_train = lgb.Dataset(xtrain, label=ytrain.values.ravel()) 
        xg_val = lgb.Dataset(xtest, label=ytest.values.ravel())

        watchlist  = [xg_train,xg_val] 

        lgbm = lgb.train(param, xg_train,ROUNDS, watchlist, early_stopping_rounds=50, verbose_eval = 10) 

        pred_lgbm = lgbm.predict(xtest)
        pred_train_lgbm[test_index] = pred_lgbm
        pred_test_lgbm = lgbm.predict(X_test_new)
        pred_test_full_lgbm += pred_test_lgbm
        
        
        
        
        logloss = log_loss(ytest, pred_lgbm) # pred pred_clf
        print ("SCORE:", logloss)  
        total += logloss
    total = total/numfolds
    print (total)

# pred_test_full = pred_test_full/5 
# pred_test_full_clf = pred_test_full_clf/5 
 pred_test_full_lgbm = pred_test_full_lgbm/5  


# Val = 0.20
## Predictions ###

# RF predictions to DF
train_pred_rf = pd.DataFrame(pred_train_rf) 
train_pred_rf.columns = ['pred_rf'] 

test_pred_rf = pd.DataFrame(pred_test_full) 
test_pred_rf.columns = ['pred_rf'] 


# MLP predictions to DF
train_pred_nn = pd.DataFrame(pred_train_clf) 
train_pred_nn.columns = ['pred_nn'] 

test_pred_nn = pd.DataFrame(pred_test_full_clf) 
test_pred_nn.columns = ['pred_nn'] 

# LightGBM predictions to DF
train_pred_lgbm = pd.DataFrame(pred_train_lgbm) 
train_pred_lgbm.columns = ['pred_lgbm'] 

test_pred_lgbm = pd.DataFrame(pred_test_full_lgbm) 
test_pred_lgbm.columns = ['pred_lgbm'] 

# Save preds
train_pred_rf.to_csv('train_rf.csv')
test_pred_rf.to_csv('test_rf.csv')

train_pred_nn.to_csv('train_nn.csv')
test_pred_nn.to_csv('test_nn.csv')

train_pred_lgbm.to_csv('train_lgbm.csv')
test_pred_lgbm.to_csv('test_lgbm.csv')




del X_train_pred['pred_rf'], X_train_pred['pred_nn'], X_train_pred['pred_lgbm'] 



X_train_preds_new = pd.DataFrame() 
X_train_preds_new = X_train_fin 
X_train_preds_new['pred_rf'] = pred_train_rf 

X_test_preds_new = pd.DataFrame() 
X_test_preds_new = X_test_fin 
X_test_preds_new['pred_rf'] = pred_test_full 


X_train_preds_new['pred_nn'] = pred_train_clf
X_test_preds_new['pred_nn'] = pred_test_full_clf

X_train_preds_new['pred_lgbm'] = pred_train_lgbm 
X_test_preds_new['pred_lgbm'] = pred_test_full_lgbm 


X_train_preds_new.to_csv('X_train_predict_final.csv', index =None)  
X_test_preds_new.to_csv('X_test_predict_final.csv', index = None) 

#X_train_pred = pd.concat([X_train_fix, train_pred_rf], axis=1,ignore_index=True) 
#X_test_pred = pd.concat([X_test_fin, test_pred_rf], axis=1)   

#X_train_pred = pd.concat([X_train_pred, train_pred_nn], axis=1) 
#X_test_pred = pd.concat([X_test_pred, test_pred_nn], axis=1)   


# X_train_pred = pd.concat([X_train_pred, train_pred_lgbm], axis=1) 
# X_test_pred = pd.concat([X_test_pred, test_pred_lgbm], axis=1)   


 Pred_prob = rf.predict_proba(X_test_new)
 
# Pred_rf = rf.predict_proba(X_test_rf)





sub = pd.DataFrame()

sub['test_id'] = test_data['test_id']
sub['is_duplicate'] = Pred_prod_df

sub.to_csv('quora_rf_v1.0.csv',index=None) 

##################################################################
### NN Classifier ###

from sklearn.neural_network import MLPClassifier 

from sklearn.cross_validation import train_test_split

 x_train, x_valid, y_train, y_valid = train_test_split(X_train_new, Y_train_fix, test_size=0.2, random_state= 4242) 



# best NN: sgd, (1000,2) act='logistic' Score = 0.2225
# best NN : adam, (600,2) act='logistic' Score = 0.2070
# best NN : adam, (300,2) act='logistic' Score = 0.2059

    rf = RandomForestClassifier(n_estimators = 200, 
                            max_depth = 11,
                            max_features = 0.3114671350604435,
                            criterion = "gini",
                            min_impurity_split = 0.0005,
                            random_state = 4242                          
                            )

    clf2 = MLPClassifier(solver='adam', alpha=1e-5,activation='logistic',
                     hidden_layer_sizes=(300, 3),
                     shuffle=True,
                     random_state=4242)


# clf.fit(x_train, y_train)  
# NN
    clf2.fit(x_train,y_train)  
# RandomForest
    rf.fit(xtrain, ytrain.values.ravel()) 






# LightGBM
xg_train = lgb.Dataset(x, label=y_train) 
xg_val = lgb.Dataset(X_val, label=y_val)
watchlist  = [xg_train,xg_val] 



pred_nn = clf2.predict_proba(x_valid)[:,1]
pred_rf = rf.predict_proba(x_valid)[:,1]
     
logloss_nn = log_loss(y_valid, pred_nn)
logloss_rf = log_loss(y_valid,pred_rf)

print ("SCORE:", logloss_nn)

  
# Predictions for NN
pred_train_nn = clf2.predict_proba(X_train_new)[:,1] 
pred_test_nn = clf2.predict_proba(X_test_new)[:,1] 

train_pred_nn = pd.DataFrame(pred_train_nn) 
train_pred_nn.columns = ['pred_nn'] 

test_pred_nn = pd.DataFrame(pred_test_nn) 
test_pred_nn.columns = ['pred_nn'] 

X_train_pred = pd.concat([X_train_fin, train_pred_nn], axis=1) 
X_test_pred = pd.concat([X_test_fin, test_pred_nn], axis=1)  

# Predictions for RandomForest





# Make predicitions
X_train_fix = Imputer().fit_transform(X_train_fix) 
X_test_fix = Imputer().fit_transform(X_test_fin) 

# Preds for stacking
X_test_new = pd.DataFrame(X_test_fix) 
X_train_new = pd.DataFrame(X_train_fix) 




p_test = clf2.predict_proba(X_test_new)[:,1]

pred_train_nn = clf2.predict_proba(X_train_new)[:,1] 
pred_test_nn = clf2.predict_proba(X_test_new)[:,1] 

train_pred_nn = pd.DataFrame(pred_train_nn) 
train_pred_nn.columns = ['pred_nn'] 

test_pred_nn = pd.DataFrame(pred_test_nn) 
test_pred_nn.columns = ['pred_nn'] 

X_train_pred = pd.concat([X_train_fin, train_pred_nn], axis=1) 
X_test_pred = pd.concat([X_test_fin, test_pred_nn], axis=1)   

sub = pd.DataFrame()
sub['test_id'] = test_data['test_id']
sub['is_duplicate'] = p_test

sub.to_csv('quora_MLP_v1.2_0.2059.csv', index=False) 



### ExtraTreeClassifier ###
























