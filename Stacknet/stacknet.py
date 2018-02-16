# -*- coding: utf-8 -*-
"""
Created on Fri May 26 15:28:17 2017

@author: Evdokimov-NN
"""

import pandas as pd

sub1 = pd.read_csv('C:/Users/Evdokimov-NN/Kaggle/Quora/Data/0.1934_lstm_199_115_0.37_0.19.csv')
sub2 = pd.read_csv('C:/Users/Evdokimov-NN/Kaggle/Quora/Data/2_xgb_lstm_v1.6.csv')

sub_fin = pd.DataFrame()

sub_fin['test_id'] = sub1['test_id']
sub_fin['is_duplicate'] = (sub1['is_duplicate'] + sub2['is_duplicate']) /2 

sub_fin.to_csv('2xgb_2lstm_try.csv',index=None) 