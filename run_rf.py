
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from typing import Dict, Tuple, Any
from imblearn.over_sampling import SMOTE, RandomOverSampler, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler
from joblib import Parallel, delayed
from itertools import product

warnings.filterwarnings("ignore")

def get_fairness_metrics(y_true: np.ndarray, y_pred: np.ndarray, groups: pd.Series, fixed_fpr: float = 0.05) -> Tuple[float, pd.DataFrame]:
    from aequitas.group import Group
    
    aequitas_df = pd.DataFrame({
        "score": y_pred,
        "label_value": y_true,
        "group": groups
    })
    
    g = Group()
    disparities_df = g.get_crosstabs(aequitas_df, score_thresholds={"score_val": [fixed_fpr]})[0]
    predictive_equality = disparities_df["fpr"].min() / disparities_df["fpr"].max()
    
    return predictive_equality, disparities_df

def evaluate(predictions: np.ndarray, ground_truth: np.ndarray, X: pd.DataFrame, fixed_fpr: float = 0.05) -> list[float]:
    fprs, tprs, thresholds = roc_curve(ground_truth, predictions)
    
    tpr = tprs[fprs < fixed_fpr][-1]
    fpr = fprs[fprs < fixed_fpr][-1]
    threshold = thresholds[fprs < fixed_fpr][-1]
    
    sorted_ages = np.sort(X["customer_age"])
    young_threshold = sorted_ages[int(0.95 * len(sorted_ages))]
    
    groups = (X["customer_age"] > young_threshold).map({True: ">young_threshold", False: "<=young_threshold"})
    
    predictive_equality, _ = get_fairness_metrics(ground_truth, predictions, groups, fixed_fpr)
    
    return [round(tpr, 10), round(predictive_equality, 6)]

def train_and_evaluate_random_forest(X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, 
                                     X_test: pd.DataFrame, y_test: pd.Series, best_params: Dict[str, Any]) -> Dict[str, Any]:
    rf_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    
    rf_model.fit(X, y)
    
    y_val_pred = rf_model.predict_proba(X_val)[:, 1]
    val_results = evaluate(y_val_pred, y_val, X=X_val)
    val_auc = roc_auc_score(y_val, y_val_pred)
    
    y_test_pred = rf_model.predict_proba(X_test)[:, 1]
    test_results = evaluate(y_test_pred, y_test, X=X_test)
    test_auc = roc_auc_score(y_test, y_test_pred)
    
    return {
        'val_results': val_results,
        'val_auc': val_auc,
        'test_results': test_results,
        'test_auc': test_auc
    }

def read_data() -> Dict[str, pd.DataFrame]:
    data = {
        'X_train': pd.read_csv("data/X_train_oh_1.csv"),
        'X_val': pd.read_csv("data/X_val_oh_1.csv"),
        'X_test': pd.read_csv("data/X_test_oh_1.csv"),
        'y_train': pd.read_csv("data/y_train_1.csv").iloc[:, 0],
        'y_val': pd.read_csv("data/y_val_1.csv").iloc[:, 0],
        'y_test': pd.read_csv("data/y_test_1.csv").iloc[:, 0],
    }
    
    return data

def apply_sampling_technique(X: pd.DataFrame, y: pd.Series, technique: str) -> Tuple[pd.DataFrame, pd.Series]:
    if technique == 'SMOTE':
        sampler = SMOTE(random_state=42)
    elif technique == 'RandomOverSampler':
        sampler = RandomOverSampler(random_state=42)
    elif technique == 'RandomUnderSampler':
        sampler = RandomUnderSampler(random_state=42)
    elif technique == 'SVMSMOTE':
        sampler = SVMSMOTE(random_state=42)
    else:
        return X, y
    
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    return X_resampled, y_resampled

def evaluate_hyperparameters(params, X_train, y_train, X_val, y_val):
    rf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_val_pred = rf.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, y_val_pred)
    return score, params

def hyperparameter_tuning(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
    param_grid = {
        'n_estimators': [600],
        'max_depth': [8, 10, 12, 14],
        'min_samples_split': [2, 4],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced'],
    }

    param_combinations = list(product(*param_grid.values()))
    param_dicts = [dict(zip(param_grid.keys(), params)) for params in param_combinations]

    results = Parallel(n_jobs=-1)(
        delayed(evaluate_hyperparameters)(params, X_train, y_train, X_val, y_val) for params in param_dicts
    )

    best_score, best_params = max(results, key=lambda x: x[0])
    return best_params, best_score

def main():
    os.makedirs('model_performance', exist_ok=True)

    data = read_data()
    
    X, y = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    
    sampling_techniques = ['None'] # , 'SMOTE', 'RandomOverSampler', 'RandomUnderSampler', 'SVMSMOTE'
    
    for technique in sampling_techniques:
        print(f"\nApplying {technique} sampling technique:")
        
        if technique != 'None':
            X_sampled, y_sampled = apply_sampling_technique(X, y, technique)
        else:
            X_sampled, y_sampled = X, y
        
        best_params, best_val_auc = hyperparameter_tuning(X_sampled, y_sampled, X_val, y_val)
        print("Best parameters found: ", best_params)
        print(f"Best validation AUC: {best_val_auc:.4f}")
        
        evaluation_results = train_and_evaluate_random_forest(
            X_sampled, y_sampled, data['X_val'], data['y_val'], 
            data['X_test'], data['y_test'], best_params
        )
        
        print("Evaluation Results:")
        print(f"Validation AUC: {evaluation_results['val_auc']:.4f}")
        print(f"Validation Evaluation: TPR at 5% FPR = {evaluation_results['val_results'][0]:.4f}, Predictive Equality = {evaluation_results['val_results'][1]:.4f}")
        print(f"Test AUC: {evaluation_results['test_auc']:.4f}")
        print(f"Test Evaluation: TPR at 5% FPR = {evaluation_results['test_results'][0]:.4f}, Predictive Equality = {evaluation_results['test_results'][1]:.4f}")

if __name__ == "__main__":
    main()
    
'''
Best parameters found:  {'n_estimators': 600, 'max_depth': 12, 'min_samples_split': 4, 'max_features': 'log2', 'class_weight': None}
Best validation AUC: 0.8666
Evaluation Results:
Validation AUC: 0.8666
Validation Evaluation: TPR at 5% FPR = 0.4591, Predictive Equality = 0.2940
Test AUC: 0.8733
Test Evaluation: TPR at 5% FPR = 0.4850, Predictive Equality = 0.2489

Applying SMOTE sampling technique:
Best parameters found:  {'n_estimators': 600, 'max_depth': 8, 'min_samples_split': 4, 'max_features': 'log2', 'class_weight': None}
Best validation AUC: 0.8552
Evaluation Results:
Validation AUC: 0.8552
Validation Evaluation: TPR at 5% FPR = 0.4291, Predictive Equality = 0.9651
Test AUC: 0.8546
Test Evaluation: TPR at 5% FPR = 0.4609, Predictive Equality = 0.9534

Applying RandomOverSampler sampling technique:
Best parameters found:  {'n_estimators': 600, 'max_depth': 12, 'min_samples_split': 2, 'max_features': 'log2', 'class_weight': None}
Best validation AUC: 0.8770
Evaluation Results:
Validation AUC: 0.8770
Validation Evaluation: TPR at 5% FPR = 0.4890, Predictive Equality = 0.9879
Test AUC: 0.8685
Test Evaluation: TPR at 5% FPR = 0.4930, Predictive Equality = 0.9890

Applying RandomUnderSampler sampling technique:
Best parameters found:  {'n_estimators': 600, 'max_depth': 12, 'min_samples_split': 4, 'max_features': 'log2', 'class_weight': None}
Best validation AUC: 0.8779
Evaluation Results:
Validation AUC: 0.8779
Validation Evaluation: TPR at 5% FPR = 0.5170, Predictive Equality = 1.0000
Test AUC: 0.8834
Test Evaluation: TPR at 5% FPR = 0.5331, Predictive Equality = 1.0000

Applying SVMSMOTE sampling technique:
Best parameters found:  {'n_estimators': 600, 'max_depth': 12, 'min_samples_split': 4, 'max_features': 'log2', 'class_weight': None}
Best validation AUC: 0.8576
Evaluation Results:
Validation AUC: 0.8576
Validation Evaluation: TPR at 5% FPR = 0.4311, Predictive Equality = 0.8059
Test AUC: 0.8573
Test Evaluation: TPR at 5% FPR = 0.4729, Predictive Equality = 0.7777

Best parameters found:  {'n_estimators': 600, 'max_depth': 10, 'min_samples_split': 2, 'max_features': 'log2', 'class_weight': 'balanced_subsample'}
Best validation AUC: 0.8778
Evaluation Results:
Validation AUC: 0.8778
Validation Evaluation: TPR at 5% FPR = 0.4870, Predictive Equality = 0.9996
Test AUC: 0.8717
Test Evaluation: TPR at 5% FPR = 0.5070, Predictive Equality = 0.9984

Best parameters found:  {'n_estimators': 600, 'max_depth': 10, 'min_samples_split': 2, 'max_features': 'log2', 'class_weight': 'balanced'}
Best validation AUC: 0.8773
Evaluation Results:
Validation AUC: 0.8773
Validation Evaluation: TPR at 5% FPR = 0.4970, Predictive Equality = 0.9996
Test AUC: 0.8729
Test Evaluation: TPR at 5% FPR = 0.5050, Predictive Equality = 0.9996
'''

'''
Findings:

RandomUnderSampler extremely good!
'''