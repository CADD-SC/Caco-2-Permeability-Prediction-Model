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
from sklearn.linear_model import LogisticRegression
from hpsklearn import HyperoptEstimator, any_preprocessing
from hpsklearn import random_forest_classifier, gradient_boosting_classifier, svc, xgboost_classification, k_neighbors_classifier
#from sklearn.svm import SVC
from hyperopt import tpe, hp
import argparse
from data_preprocessing import data_preprocessing

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='Caco2_permeability')
    parser.add_argument('--validation', action='store_true', default=False)
    parser.add_argument('--prediction', action='store_true', default=False)
    parser.add_argument('--single_mode', action='store_true', default=False)

    args  =parser.parse_args()
    
    features = eval(open(f'./features.txt', 'r').read())
    test_data = pd.read_csv(args.file_name)

    if args.prediction :#w/o_bioclass
        print(f'prediction step for {args.model_name}, data set for {args.file_name} ({len(test_data)})')
        # prprocessing data feature
        test_data['bioclass']=1
        scaled_data = data_preprocessing(test_data)
        result_df = pd.DataFrame({'SMILES':scaled_data['SMILES']})
        # load trained model
        # learner_model = pickle.load(open(f'./best_models_v2/{args.model_name}.pkl', 'rb'))
        learner_model = pickle.load(open(f'{args.model_name}.pkl', 'rb'))
        # predict label (int)
        predicted = learner_model.predict(scaled_data[features].values)
        result_df[f'{args.model_name}']=learner_model.predict(scaled_data[features].values)
        # predict prediction prob (float)
        predict_proba = learner_model.predict_proba(scaled_data[features].values)
        max_proba = np.max(predict_proba, axis=1)
        result_df['pred_prob']=max_proba
        result_df.to_csv(f'{args.model_name}_output.csv', index=False)

    elif args.validation : #w_bioclass
        print(f'validation step for {args.model_name}, data set for {args.file_name} ({len(test_data)})')

        # prprocessing data feature
        scaled_data = data_preprocessing(test_data)

        # load trained model
        # learner_model = pickle.load(open(f'./best_models_v2/{args.model_name}.pkl', 'rb'))
        learner_model = pickle.load(open(f'{args.model_name}.pkl', 'rb'))

        # predict label (int)
        predicted = learner_model.predict(scaled_data[features].values)
        tn, fp, fn, tp = confusion_matrix(scaled_data['bioclass'], predicted).ravel()
        print('Se:', tp/(tp+fn))
        print('Sp:', tn/(tn+fp))
        print('acc:', (tp+tn)/(tp+fp+tn+fn))
        print('mcc:', ((tp*tn)-(fp*fn))/((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5 )

        # predict prediction prob (float)
        predict_proba = learner_model.predict_proba(scaled_data[features].values)
        print('auc:', roc_auc_score(scaled_data['bioclass'], predict_proba[:,1]) )

    else:
        print('choose mode [validation, prediction]')
