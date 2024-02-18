import sys
sys.path.append(r'C:\Users\lizhiwei\Desktop\HMM_XGB_Doc')

from data_code import form_train_test, form_label, train_combine, test_combine
from HMM_XGB import HMM_XGB
from pred_prob_HMM_XGB import pred_prob_HMM_XGB
from return_back_test import return_back_test
import numpy as np
from hmmlearn import hmm


feature_col = ['DEMA']
start_date = '2022/01/01'
date_split = '2022/12/31'

train, test = form_train_test(feature_col, start_date, date_split)

n_states = 5
# A, xgb_model, pi = HMM_XGB(train, n_states, True)

# pred_prob_XGB_train = pred_prob_HMM_XGB(A, xgb_model, pi, train)
# pred_state_XGB_train = np.argmax(pred_prob_XGB_train, axis = 1)

# pred_prob_XGB_test = pred_prob_HMM_XGB(A, xgb_model, pi, test)
# pred_state_XGB_test = np.argmax(pred_prob_XGB_test, axis = 1)


# states = []
# states.append(pred_state_XGB_train)
# states.append(pred_state_XGB_test)
# return_back_test(states, n_states, 'te')


states = []
model = hmm.GaussianHMM(n_components = n_states, covariance_type = 'diag', n_iter = 1000)
model.fit(train)
states.append(model.predict(train))
states.append(model.predict(test))



# return_back_test(states, n_states, 'te')



# ######################### LSTM Model #####################################

# feature_cols = [['WMA'], ['HT_TRENDLINE'], ['EMA12'], ['T3']]

# start_date = '2022/01/01'
# date_split = '2022/12/31'

# pred_prob_tr = []
# pred_prob_te = []

# label = form_label(start_date, date_split, 0.02, 6)

# for i in feature_cols :
#     train, test = form_train_test(i, start_date, date_split)

#     n_states = 5
#     A, xgb_model, pi = HMM_XGB(train, n_states, True)
    
#     pred_prob_XGB_train = pred_prob_HMM_XGB(A, xgb_model, pi, train)
#     pred_prob_XGB_test = pred_prob_HMM_XGB(A, xgb_model, pi, test)
    
#     pred_prob_tr.append(pred_prob_XGB_train)
#     pred_prob_te.append(pred_prob_XGB_test)

# final_train_X, final_train_Y = train_combine(pred_prob_tr, label)
# final_test_X = test_combine(pred_prob_te)

# self_LSTM(final_train_X, final_train_Y)






