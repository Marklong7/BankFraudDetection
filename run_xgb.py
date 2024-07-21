import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from typing import Dict, Tuple, Any
from imblearn.over_sampling import SMOTE, RandomOverSampler, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler

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

def train_and_evaluate_xgboost(X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, 
                               X_test: pd.DataFrame, y_test: pd.Series, best_params: Dict[str, Any]) -> Dict[str, Any]:
    xgb_model = xgb.XGBClassifier(**best_params, use_label_encoder=False)
    
    xgb_model.fit(X, y)
    
    y_train_pred = xgb_model.predict_proba(X)[:, 1]
    train_auc = roc_auc_score(y, y_train_pred)
    
    y_val_pred = xgb_model.predict_proba(X_val)[:, 1]
    val_results = evaluate(y_val_pred, y_val, X=X_val)
    
    y_test_pred = xgb_model.predict_proba(X_test)[:, 1]
    test_results = evaluate(y_test_pred, y_test, X=X_test)
    
    return {
        'train_auc': train_auc,
        'val_results': val_results,
        'test_results': test_results
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

def main():
    os.makedirs('model_performance', exist_ok=True)

    data = read_data()
    
    X, y = data['X_train'], data['y_train']
    
    sampling_techniques = ['None'] # 'None', 'RandomUnderSampler', 'SVMSMOTE'
    
    for technique in sampling_techniques:
        print(f"\nApplying {technique} sampling technique:")
        
        if technique != 'None':
            X_sampled, y_sampled = apply_sampling_technique(X, y, technique)
        else:
            X_sampled, y_sampled = X, y
        
        xgb_model = xgb.XGBClassifier()

        param_grid = {
            'n_estimators': [300, 400],
            'learning_rate': [0.1],
            'max_depth': [3, 4],
            #'subsample': [0.9, 1],
            #'colsample_bytree': [0.8, 1.0],
            #'reg_alpha': [0.5, 0.7],
            'scale_pos_weight': [103], # use balanced weight
            # The scale_pos_weight value is used to scale the gradient for the positive class.
            # Use 100 can simulate the effect that we use class_weight = 'balanced' in sklearn logistics.
        }
        
        kf = KFold(n_splits=4, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='roc_auc', cv=kf, n_jobs=-1)
        
        grid_search.fit(X_sampled, y_sampled)
        
        best_params = grid_search.best_params_
        print("Best parameters found: ", best_params)
        
        evaluation_results = train_and_evaluate_xgboost(
            X_sampled, y_sampled, data['X_val'], data['y_val'], 
            data['X_test'], data['y_test'], best_params
        )
        
        print("Evaluation Results: ", evaluation_results)

if __name__ == "__main__":
    main()
    
'''
Model results:

    !!!Best Model without re-sampling!!!
Best parameters found:  {'colsample_bytree': 0.6, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 400, 'reg_alpha': 0.7, 'reg_lambda': 1, 'subsample': 0.9}
Evaluation Results:  {'train_auc': 0.9206953696160034, 'val_results': [0.5209580838, 0.410855], 'test_results': [0.5731462926, 0.279965]}

    !!!Re-weight!!!
Applying None sampling technique:
Best parameters found:  {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 300, 'scale_pos_weight': 100}
Evaluation Results:  {'train_auc': 0.9261283536630169, 'val_results': [0.5309381238, 0.90646], 'test_results': [0.5551102204, 0.865386]}

Applying RandomOverSampler sampling technique:
Best parameters found:  {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 400, 'reg_alpha': 0.7, 'subsample': 0.9}
Evaluation Results:  {'train_auc': 0.9652418469217535, 'val_results': [0.49500998, 0.854023], 'test_results': [0.5130260521, 0.733977]}

    !!!Best Model with re-sampling!!!
Applying RandomUnderSampler sampling technique:
Best parameters found:  {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 300, 'reg_alpha': 0.7, 'subsample': 0.9}
Evaluation Results:  {'train_auc': 0.9501389258884331, 'val_results': [0.4930139721, 0.918649], 'test_results': [0.5170340681, 0.898675]}

Applying SMOTE sampling technique:
Best parameters found:  {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 400, 'reg_alpha': 0.7, 'subsample': 0.9}
Evaluation Results:  {'train_auc': 0.9989708923658015, 'val_results': [0.4810379242, 0.513641], 'test_results': [0.5190380762, 0.371253]}

Applying SVMSMOTE sampling technique:
Best parameters found:  {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 400, 'reg_alpha': 0.5, 'subsample': 0.9}
Evaluation Results:  {'train_auc': 0.9983936148780871, 'val_results': [0.5149700599, 0.447515], 'test_results': [0.5531062124, 0.345252]}
'''

'''
Findings

'''

'''

'''

