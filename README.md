# HMM-XGB-Strategy

*This is the HMM-XGB strategy from my internship in Taoli. It contains the whole Gaussian-HMM with XGBoost training and testing code.*

- Obtaining the data (using IH for example)
- Using HMM to predict the hidden states of each data (remember change the number of hidden states)
- Using XBGoost to retraining and repredicting, updating the hidden states if the log-likelihood improve.
- Select time area to slide update model.


## Using train_HMM_XGB_model.py to run the entire project.

*Pay attention to project location, data location, running time, training or testing.* 

*(Some parameters and hyper-parameters must be changed in own situation.)*

For example, the hidden states, the hyper-parameters in XGBoost. And the raw_da.csv is an outdated example.


*Actually, I also tried LSTM to exchange the XGBoost, but it needs more training time and data form changes(sequence form).*
