import pickle
import pandas as pd
import json

def predict_mpg(config):
    ##loading the model from the saved file
    pkl_filename = "model.pkl"
    with open(pkl_filename, 'rb') as f_in:
        model = pickle.load(f_in)

    if type(config) == dict:
        df = pd.DataFrame(config)
    else:
        df = config
    
    y_pred_1 = model.check_rule_001(df)
    y_pred_2 = model.check_rule_002(df, df['entityId'])

    
    if y_pred_1 == True:
        return 'fraud'
    elif y_pred_2 == True:
        return 'fraud'
    else :
        #condtion 3 and 4 from ml model
        return ''
    