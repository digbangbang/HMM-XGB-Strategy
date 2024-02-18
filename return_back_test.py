import matplotlib.pyplot as plt
import numpy as np
from data_code import form_backtest_da


def return_back_test(states, n_states, te_or_tr) :
    start_date = '2022/01/01'
    date_split = '2022/12/31'
    
    bt_train, bt_test = form_backtest_da(start_date, date_split)
    
    bt_train['Log_return'] = np.log(bt_train['Close'] / bt_train['Close'].shift(1))
    bt_test['Log_return'] = np.log(bt_test['Close'] / bt_test['Close'].shift(1))
    bt_train['state'] = states[0]
    bt_test['state'] = states[1]
    
    if te_or_tr == 'te' :
        for j in range(n_states):
            state = (bt_test.loc[:,'state'] == j)
            idx = np.append(0,state[:-1])
            bt_test['{}_State_Return'.format(j)] = np.exp(bt_test.Log_return.multiply(idx, axis = 0).cumsum())
            plt.plot(bt_test['date'], bt_test['{}_State_Return'.format(j)])
    elif te_or_tr == 'tr' :
        for j in range(n_states):
            state = (bt_train.loc[:,'state'] == j)
            idx = np.append(0,state[:-1])
            bt_train['{}_State_Return'.format(j)] = np.exp(bt_train.Log_return.multiply(idx, axis = 0).cumsum())
            plt.plot(bt_train['date'], bt_train['{}_State_Return'.format(j)])