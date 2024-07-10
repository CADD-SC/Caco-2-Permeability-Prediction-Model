import time
import pandas as pd
import numpy as np
from numpy import array
import pickle
import os
from os import path
import sklearn
from sklearn.metrics import roc_auc_score, make_scorer, recall_score, accuracy_score, matthews_corrcoef, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate

from hpsklearn import HyperoptEstimator, any_preprocessing
from hpsklearn import random_forest_classifier, gradient_boosting_classifier, svc, xgboost_classification, k_neighbors_classifier
from hyperopt import tpe, hp

import argparse
from data_preprocessing import data_preprocessing

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, required=True, help='Input file name for predictions or validation.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file (.pkl) to be used for predictions.')
    parser.add_argument('--validation', action='store_true', default=False, help='Run in validation mode.')
    parser.add_argument('--prediction', action='store_true', default=False, help='Run in prediction mode.')

    args = parser.parse_args()
    
    features = eval(open(f'./features.txt', 'r').read())

    if args.prediction: # Prediction mode without bioclass
        print('Prediction mode')
        test_data = pd.read_csv(args.file_name)
        test_data['bioclass'] = 1
        scaled_data = data_preprocessing(test_data)

        print('Size of data set: ', len(test_data))
        print(f'Using model: {args.model_path}')
        learner_model = pickle.load(open(args.model_path, 'rb'))
        predicted = learner_model.predict(scaled_data[features].values)
        test_data['prediction'] = predicted
        output_file = args.file_name.replace('.csv', '_predictions.csv')
        test_data.to_csv(output_file, index=False)
        print(f'Saved output in {output_file}')

    elif args.validation: # Validation mode with bioclass
        print('Validation mode')
        test_data = pd.read_csv(args.file_name)
        scaled_data = data_preprocessing(test_data)

        print('Size of data set: ', len(test_data))
        print(f'Using model: {args.model_path}')
        learner_model = pickle.load(open(args.model_path, 'rb'))
        print(type(learner_model))

        predicted = learner_model.predict(scaled_data[features].values)
        tn, fp, fn, tp = confusion_matrix(scaled_data['bioclass'], predicted).ravel()
        print('Se:', tp/(tp+fn))
        print('Sp:', tn/(tn+fp))
        print('Acc:', (tp+tn)/(tp+fp+tn+fn))
        print('MCC:', ((tp*tn)-(fp*fn))/((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5)

        try:
            predicted_proba = learner_model.predict_proba(scaled_data[features].values)[:, 1]
        except:
            predicted_proba = learner_model.decision_function(scaled_data[features].values)
        print('AUC:', roc_auc_score(scaled_data['bioclass'], predicted_proba))
    else:
        print('Please choose a mode: [--validation or --prediction]')

