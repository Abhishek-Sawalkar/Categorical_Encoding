import os
import numpy as np
import pandas as pd 
from sklearn import ensemble 
from sklearn import preprocessing
from sklearn import metrics

import joblib

from . import dispatcher

TEST_DATA = os.environ.get("TEST_DATA")
MODEL = os.environ.get("MODEL")


def predict():
    df=pd.read_csv(TEST_DATA)
    test_idx=df.id.values
    df = df.drop('id', axis=1)

    predictions=None 

    for FOLD in range(5):

        df=pd.read_csv(TEST_DATA)
        df = df.drop('id', axis=1)

        encoder = joblib.load(os.path.join("Categorical_Encoding/models", f"{MODEL}_{FOLD}_label_encoder.pkl"))
        
        for c in df.columns:
            lbl=encoder[c]
            df.loc[:, c] = lbl.transform(df[c].tolist())

        # data is ready to train
        model = joblib.load(os.path.join("Categorical_Encoding/models", f"{MODEL}_{FOLD}.pkl"))
        preds = model.predict_proba(df)[:, 1]

        if FOLD == 0:
            predictions = preds
        else:
            predictions = predictions + preds
    
    predictions=predictions/5

    sub = pd.DataFrame(np.column_stack((test_idx, predictions)), columns=['id', 'target'])
    return sub




if __name__ == "__main__":
    submission = predict()
    submission['id']= submission['id'].astype(int)
    submission.to_csv("Categorical_Encoding/models/submission.csv", index=False)