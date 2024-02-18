import xgboost as xgb

def form_B_matrix_by_XGB(model, O, pi) :
    
    pred = model.predict(xgb.DMatrix(O))
    
    B = pred / pi
    
    return B