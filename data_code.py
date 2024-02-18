from SQLData import GetData, TLData
import pandas as pd
import numpy as np



def form_train_test(feature_col, start_date, date_split) :
    
    #feature_col = ['EMA12', 'ATAN', 'DEMA', 'MIDPOINT', 'Open']
    #date_split = '2022/12/31'
    gt = GetData()
    data = gt.Future_hist_Mcandle('IH', '2015.01.01', '2023.08.10', types = 'main', ktype = 60)
    df = gt.Indicators(data)
    df[['date', 'symbol']] = data[['date', 'main_symbol']]
    data_train = df[(df['date'] <= pd.Timestamp(date_split)) & (df['date'] >= pd.Timestamp(start_date))]
    data_test = df[df['date'] > pd.Timestamp(date_split)]
    
    data_train = data_train[feature_col]
    data_test = data_test[feature_col]
    
    for i in range(data_train.shape[1]) :
        data_test.iloc[:,i].replace(np.inf, 2 * np.max(data_test.iloc[:,i]), inplace = True)
        data_test.iloc[:,i].replace(-np.inf, 2 * np.min(data_test.iloc[:,i]), inplace = True)
        data_train.iloc[:,i].replace(np.inf, 2 * np.max(data_train.iloc[:,i]), inplace = True)
        data_train.iloc[:,i].replace(-np.inf, 2 * np.min(data_train.iloc[:,i]), inplace = True)
        
        # data_test.iloc[:,i].replace(np.nan, np.average(data_test.iloc[:,i]), inplace = True)
        # data_train.iloc[:,i].replace(np.nan, 2 * np.max(data_train.iloc[:,i]), inplace = True)
        
    data_train.dropna(axis = 0, inplace = True)
    data_test.dropna(axis = 0, inplace = True)
    
    return data_train, data_test

def form_backtest_da(start_date, date_split) :
    
    gt = GetData()
    data = gt.Future_hist_Mcandle('IH', '2015.01.01', '2023.08.10', types = 'main', ktype = 60)
    df = gt.Indicators(data)
    df[['date', 'symbol']] = data[['date', 'main_symbol']]
    data_train_bt = df[(df['date'] <= pd.Timestamp(date_split)) & (df['date'] >= pd.Timestamp(start_date))]
    data_test_bt = df[df['date'] > pd.Timestamp(date_split)]
    
    feature_col = ['date', 'symbol', 'Close']
    
    data_train_bt = data_train_bt[feature_col]
    data_test_bt = data_test_bt[feature_col]
    
        
    data_train_bt.dropna(axis = 0, inplace = True)
    data_test_bt.dropna(axis = 0, inplace = True)
    
    return data_train_bt, data_test_bt


def form_label(start_date, date_split, threshold, T) : # T = 5 * 6 

    # 0 到达竖栏杆，1 到达上栏杆，-1 到达下栏杆，-2 长度不够    
    gt = GetData()
    data = gt.Future_hist_Mcandle('IH', '2015.01.01', '2023.08.10', types = 'main', ktype = 60)
    df = gt.Indicators(data)
    df[['date', 'symbol']] = data[['date', 'main_symbol']]
    df = df[(df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(date_split))]
    
    close_price = df['Close'].values
    label = np.zeros(len(close_price)) - 2
    
    for i in range(len(label)) :
        if len(label) - i - 1 < T :
            continue
        else :
            now_close = close_price[i]
            temp_threshold = now_close * threshold
            
            flag = 0
            for j in range(T) :
                if close_price[i + j + 1] - now_close > temp_threshold :
                    label[i] = 1
                    flag = 1
                    break
                elif close_price[i + j + 1] - now_close < - temp_threshold :
                    label[i] = -1
                    flag = 1
                    break
            if flag == 0 :
                label[i] = 0
    
    return label
    
def train_combine(X, label) :
    
    matrix_X = np.zeros((len(label), 0))
    for i in range(len(X)) :
        matrix_X = np.column_stack((matrix_X, X[i]))
    
    temp = label != -2
    result_X = matrix_X[temp]
    label = label[temp]
    
    return result_X, label

def test_combine(X) :
    matrix_X = np.zeros((len(X[0]), 0))
    for i in range(len(X)) :
        matrix_X = np.column_stack((matrix_X, X[i]))
    
    return matrix_X