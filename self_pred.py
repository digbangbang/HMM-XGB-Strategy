import numpy as np

def self_pred(B, A, pi) :
    
    log_likelihood = 0
    n_states = len(pi)
    init_flag = 1
    
    T_all = B.shape[0]
    now_B = B
    now_state = np.zeros(T_all)
    now_state_prob = np.zeros((T_all, n_states))
    # scale_all = np.zeros(T_all)
    
    for i in range(T_all) :
        
        if i == 0 :
            now_state_prob[i] = now_B[i] * pi
            # scale_all[0] = sum(now_state_prob[i])
        else :
            for k in range(n_states) :
                temp = now_state_prob[i - 1] * A[:, k] * now_B[i, k]
                now_state_prob[i, k] = max(temp)
        # scale_all[i] = 1 / sum(now_state_prob[i])
        now_state_prob[i] = now_state_prob[i] / np.sum(now_state_prob[i])
        now_state[i] = np.argmax(now_state_prob[i])
        
        
    for i in range(T_all) :
        if i == 0 :
            now_log_likelihood = np.log(pi[int(now_state[i])]) + np.log(now_B[i, int(now_state[i])])
        else :
            now_log_likelihood += np.log(A[int(now_state[i-1]), int(now_state[i])]) + np.log(now_B[i, int(now_state[i])])
    
    if init_flag == 1 :
        state = now_state
        state_prob = now_state_prob
        init_flag = 0
    
    log_likelihood = now_log_likelihood
    
    
    
    return state, state_prob, log_likelihood