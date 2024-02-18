import numpy as np

def re_estimate(A, B_all, pi) :
    
    n_states = B_all.shape[1]
    T_all = B_all.shape[0]
    alpha_all = np.zeros((T_all, n_states))
    beta_all = np.zeros((T_all, n_states))
    di_gamma_all = np.zeros((T_all, n_states, n_states))
    gamma_all = np.zeros((T_all, n_states))
    scale_all = np.zeros(T_all)
    
    
    # : compute alpha 向前概率
    for i in range(n_states) :
        alpha_all[0, i] = pi[i] * B_all[0, i]
    scale_all[0] = sum(alpha_all[0])
    
    for t in range(1, T_all) :
        for i in range(n_states) :
            alpha_all[t, i] = 0
            for j in range(n_states) :
                alpha_all[t, i] += alpha_all[t - 1, j] * A[j, i]
            alpha_all[t, i] = alpha_all[t, i] * B_all[t, i]
        scale_all[t] = 1 / sum(alpha_all[t])
        alpha_all[t] = alpha_all[t] * scale_all[t]
        
    
    # : compute beta 向后概率
    beta_all[T_all - 1] = scale_all[T_all - 1]
    
    for t in range(T_all - 2, - 1, - 1) :
        for i in range(n_states) :
            beta_all[t, i] = 0
            for j in range(n_states) :
                beta_all[t, i] += A[i, j] * B_all[t + 1, j] * beta_all[t + 1, j]
            beta_all[t, i] = scale_all[t] * beta_all[t, i]


    # : compute di_gamma and gamma
    for t in range(T_all - 1) :
        for i in range(n_states) :
            gamma_all[t, i] = 0
            for j in range(n_states) :
                di_gamma_all[t, i, j] = alpha_all[t, i] * A[i, j] * B_all[t + 1, j] * beta_all[t + 1, j]
                gamma_all[t, i] += di_gamma_all[t, i, j]
    t = T_all - 1
    gamma_all[t] = alpha_all[t]
    
   
    
    
    # : re_estimate A
    for i in range(n_states) :
        for j in range(n_states) :
            numer = np.sum(di_gamma_all[0: T_all - 1, i, j])
            denom = np.sum(gamma_all[0 : T_all - 1, i])
            A[i, j] = numer / denom
        
            
    return A, gamma_all
        
    
    
    