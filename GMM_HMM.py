from hmmlearn import hmm

def GMM_HMM(O, n_states, v_type, n_iter, verbose) :
    
    model = hmm.GaussianHMM(n_components = n_states, covariance_type = v_type, n_iter = n_iter, verbose = verbose).fit(O)
    
    A = model.transmat_ # : 转移概率矩阵
    
    _, S = model.decode(O, algorithm = 'viterbi') # : 解码状态
    gamma = model.predict_proba(O) # : 状态概率
    
    return A, S, gamma