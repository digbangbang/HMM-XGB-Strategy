import numpy as np
from form_B_matrix_by_XGB import form_B_matrix_by_XGB
from self_pred import self_pred

def pred_prob_HMM_XGB(A, model, pi, O) :
    n_states = len(pi)
    pred_prob = np.zeros((O.shape[0], n_states))
    
    now_B = form_B_matrix_by_XGB(model, O, pi)
    _, now_pred_proba, _ = self_pred(now_B, A, pi)
    
    pred_prob = now_pred_proba
    
    return pred_prob