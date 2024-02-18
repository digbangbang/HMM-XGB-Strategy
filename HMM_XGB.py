from GMM_HMM import GMM_HMM
import numpy as np
from re_estimate import re_estimate
from self_pred import self_pred
from self_xgb import self_xgb
from form_B_matrix_by_XGB import form_B_matrix_by_XGB



def HMM_XGB(O, n_states, verbose = True) :

    iteration = 0
    stop_flag = 0
    log_likelihood = - np.inf
    min_delta = 1e-4
    
    A, S, gamma = GMM_HMM(O, n_states, 'diag', 1000, True)
    prior_pi = np.array([sum(S == i) / len(S) for i in range(n_states)])
    model = 1
    B_Matrix = gamma / prior_pi
    
    record_log_likelihood = []
    best_result = []
    
    
    while stop_flag <= 3 and iteration <= 20:
        A, gamma = re_estimate(A, B_Matrix, prior_pi)
        
        model = self_xgb(O, gamma, n_states)
        
        B_Matrix = form_B_matrix_by_XGB(model, O, prior_pi)
        
        new_S, _, new_log_likelihood = self_pred(B_Matrix, A, prior_pi)
        
        record_log_likelihood.append(new_log_likelihood)
        
        if len(best_result) == 0 :
            best_result = [A, model, prior_pi, new_log_likelihood]
            temp = gamma
        elif new_log_likelihood > best_result[3] :
            best_result = [A, model, prior_pi, new_log_likelihood]
            temp = gamma
        
        if new_log_likelihood - log_likelihood <= min_delta :
            stop_flag += 1
        else :
            stop_flag = 0
            
        log_likelihood = new_log_likelihood
        print(log_likelihood)
        
        iteration += 1
        
        
    model = self_xgb(O, temp, n_states)
    best_result[1] = model
    
    return best_result[0], best_result[1], best_result[2]






