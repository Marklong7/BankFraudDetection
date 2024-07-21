import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, confusion_matrix, roc_curve, roc_auc_score
# from sklearn.model_selection import cross_val_score, KFold
from aequitas.group import Group
import warnings
from joblib import Parallel, delayed
import os
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE, SVMSMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.combine import SMOTETomek

warnings.filterwarnings("ignore")

def Plot_ROC(fpr, tpr):
    plt.plot(fpr, tpr, label = 'ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
    
def Get_fairness_metrics(y_true, y_pred, groups, FIXED_FPR = 0.05):
    # aequitas.group
    g = Group()
    aequitas_df = pd.DataFrame(
        {"score": y_pred,
         "label_value": y_true,
         "group": groups}
    )
    # Use aequitas to compute confusion matrix metrics for every group.
    disparities_df = g.get_crosstabs(aequitas_df, score_thresholds={"score_val": [FIXED_FPR]})[0]
    #print("This is the disparities df:")
    #display(disparities_df)
    predictive_equality = disparities_df["fpr"].min() / disparities_df["fpr"].max()
    return predictive_equality, disparities_df

def Evaluation(predictions, ground_truth, X, FIXED_FPR=0.05):
    result = []
    fprs, tprs, thresholds = roc_curve(ground_truth, predictions)
    
    # Identify the threshold for the given FPR
    tpr = tprs[fprs < FIXED_FPR][-1]
    fpr = fprs[fprs < FIXED_FPR][-1]
    threshold = thresholds[fprs < FIXED_FPR][-1]
    
    # Determine the new threshold for age to maintain 95% and 5% group sizes
    sorted_ages = np.sort(X["customer_age"])
    young_threshold = sorted_ages[int(0.95 * len(sorted_ages))]  # Threshold for the younger group (95%)
    
    # Define groups based on the new threshold
    groups = (X["customer_age"] > young_threshold).map({True: ">young_threshold", False: "<=young_threshold"})
    
    # Calculate fairness metrics
    predictive_equality, disparities_df = Get_fairness_metrics(ground_truth, predictions, groups, FIXED_FPR)
    
    # Print and store results
    print(f"TPR under the threshold: {round(tpr, 6)}")
    
    result.append(round(tpr, 10))
    result.append(round(predictive_equality, 6))
    
    return result

def train_and_evaluate_single_model(X, y, X_val, y_val, X_test, y_test, model_params):
    """
    Trains and evaluates a single logistic regression model using AUC.

    Parameters:
    - X: Training features
    - y: Training labels
    - X_val: Validation features
    - y_val: Validation labels
    - X_test: Testing features
    - y_test: Testing labels
    - model_params: Dictionary of parameters to pass to the LogisticRegression model

    Returns:
    - Evaluation results for validation and test sets, and AUC score on training set
    """
    warnings.filterwarnings("ignore")
    #print("A model is being trained")
    lr_model = LogisticRegression(**model_params, n_jobs=-1, max_iter=1000)

    # Train on full training set
    lr_model.fit(X, y)
    #print("The training is done")

    # Calculate AUC on training set
    y_train_pred = lr_model.predict_proba(X)[:, 1]
    train_auc = roc_auc_score(y, y_train_pred)
    
    # Evaluate on validation set
    y_val_pred = lr_model.predict_proba(X_val)[:, 1]
    val_results = Evaluation(y_val_pred, y_val, X=X_val)
    
    # Evaluate on test set
    y_test_pred = lr_model.predict_proba(X_test)[:, 1]
    test_results = Evaluation(y_test_pred, y_test, X=X_test)
    
    return {
        'train_auc': train_auc,
        'val_results': val_results,
        'test_results': test_results
    }

def train_evaluate_logistic_regression_parallel(X, y, X_val, y_val, X_test, y_test, model_params_list):
    """
    Trains and evaluates logistic regression models in parallel.

    Parameters:
    - X: Training features
    - y: Training labels
    - X_val: Validation features
    - y_val: Validation labels
    - X_test: Testing features
    - y_test: Testing labels
    - model_params_list: List of dictionaries, each containing parameters for a LogisticRegression model

    Returns:
    - List of evaluation results for each model
    """
    warnings.filterwarnings("ignore")
    results = Parallel(n_jobs=-1)(delayed(train_and_evaluate_single_model)(X, y, X_val, y_val, X_test, y_test, model_params) for model_params in model_params_list)
    return results

def log_transform(df, features):
    df = df.copy()
    for feature in features:
        df[feature] = df[feature].apply(lambda x: np.log1p(x) if x >= 0 else x)
    return df

def scale_data(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert scaled arrays back to DataFrames with original column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    return X_train_scaled, X_val_scaled, X_test_scaled

def print_and_save_results(results, model_params_list, prefix=''):
    for i, result in enumerate(results):
        model_info = (
            f"Model {i+1} - Params: {model_params_list[i]}\n"
            f"Training AUC: {result['train_auc']:.6f}\n"
            f"Validation TPR: {result['val_results'][0]:.6f}, Predictive Equality: {result['val_results'][1]:.6f}\n"
            f"Test TPR: {result['test_results'][0]:.6f}, Predictive Equality: {result['test_results'][1]:.6f}\n"
            "--------------------------------------------------------------\n"
        )
        print(model_info)
        with open(f'model_performance/logistic_{prefix}{i+1}.txt', 'w') as f:
            f.write(model_info)

def read_data():
    data = {
        'X_train': pd.read_csv("data/X_train_oh_1.csv"),
        'X_val': pd.read_csv("data/X_val_oh_1.csv"),
        'X_test': pd.read_csv("data/X_test_oh_1.csv"),
        'y_train': pd.read_csv("data/y_train_1.csv").iloc[:, 0],
        'y_val': pd.read_csv("data/y_val_1.csv").iloc[:, 0],
        'y_test': pd.read_csv("data/y_test_1.csv").iloc[:, 0],  # ravel()
    }
    
    return data     
            
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    os.makedirs('model_performance', exist_ok=True)

    data = read_data()
    # Log-transformed data evaluation
    print("Applying log transformation on right-skewed features")
    
    '''
    y = [1, 1, 1, 1, 0, 0]
    compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
    array([1.5 , 0.75])
    '''
    model_params_list = [
        #{'C': 0.0005, 'solver': 'newton-cholesky', 'class_weight': 'balanced'},
        #{'C': 0.001, 'solver': 'newton-cholesky', 'class_weight': 'balanced'},
        #{'C': 0.002, 'solver': 'newton-cholesky', 'class_weight': 'balanced'},
        #{'C': 0.015, 'solver': 'newton-cholesky'},
        #{'C': 0.01, 'solver': 'newton-cholesky'},
        {'C': 0.005, 'solver': 'newton-cholesky'},
    ]
        
    features_to_log_transform = [
        'prev_address_months_count', 'days_since_request', 'intended_balcon_amount', 
        'bank_branch_count_8w', 'bank_months_count', 'session_length_in_minutes'
    ]
    
    for dataset in ['X_train', 'X_val', 'X_test']:
        data[f'{dataset}_transformed'] = log_transform(data[dataset], features_to_log_transform)

    evaluation_results_transformed = train_evaluate_logistic_regression_parallel(
        data['X_train_transformed'], data['y_train'], data['X_val_transformed'], data['y_val'], 
        data['X_test_transformed'], data['y_test'], model_params_list
    )
    print_and_save_results(evaluation_results_transformed, model_params_list, 'log_transformed_')


    '''
    Observation 1:
        when max_iter = 200/1000,
        log-transformation always works, for both performance and fairness.
    
    Observation 2:
        when max_iter = 200/1000,
        After applying z-score normalization, 
            -> almost all optimizers have exactly the same results.
            -> we can use more optimizers, but the performance is worse than the best model with log transformation.
            
    Observation 3:
        re-sampling method can increase the Predictive Equality significantly.
    '''


    # Define resampling methods
    resampling_methods = {
        #'SMOTE': SMOTE(random_state=42),
        #'NearMiss': NearMiss(),
        'RandomOverSampler': RandomOverSampler(random_state=523),
        'RandomUnderSampler': RandomUnderSampler(random_state=523),
        #'SVMSMOTE': SVMSMOTE(random_state=523),
        #'SMOTETomek': SMOTETomek(random_state=42)
    }
    
    model_params_list = [
        {'C': 0.5, 'solver': 'newton-cholesky'},
        {'C': 1, 'solver': 'newton-cholesky'},
        {'C': 1.5, 'solver': 'newton-cholesky'},
        {'C': 3, 'solver': 'newton-cholesky'},
    ]
    for resampling_name, resampler in resampling_methods.items():
        print(f"\nApplying {resampling_name}")
    
        if resampler is not None:
            X_train_resampled, y_train_resampled = resampler.fit_resample(data['X_train_transformed'], data['y_train'])
        else:
            X_train_resampled, y_train_resampled = data['X_train_transformed'], data['y_train']
    
        evaluation_results = train_evaluate_logistic_regression_parallel(
            X_train_resampled, y_train_resampled, 
            data['X_val_transformed'], data['y_val'], 
            data['X_test_transformed'], data['y_test'], 
            model_params_list
        )
    
        print_and_save_results(evaluation_results, model_params_list, f'log_transformed_{resampling_name}_')
        
        
    '''
    Model Performance:
    
!!!Best Model with log-transformation!!!
    *** Model 1 - Params: {'C': 0.01, 'solver': 'newton-cholesky'}
    Training AUC: 0.864543
    Validation TPR: 0.510978, Predictive Equality: 0.314580
    Test TPR: 0.549098, Predictive Equality: 0.223930
    
!!!Best Model with both log-transformation and balanced class weight!!!
    Model 2 - Params: {'C': 0.0005, 'solver': 'newton-cholesky', 'class_weight': 'balanced'}
    Training AUC: 0.865425
    Validation TPR: 0.504990, Predictive Equality: 0.957802
    Test TPR: 0.537074, Predictive Equality: 0.943861
    
    Model 3 - RandomOverSampler(42) - Params: {'C': 3, 'solver': 'newton-cholesky'} 
    Training AUC: 0.868223
    Validation TPR: 0.504990, Predictive Equality: 0.917609
    Test TPR: 0.531062, Predictive Equality: 0.891563

!!!Best Model with both log-transformation and re-sampling technique!!!
    *** Model 4 - RandomUnderSampler(42) - Params: {'C': 1, 'solver': 'newton-cholesky'}
    Training AUC: 0.861346
    Validation TPR: 0.510978, Predictive Equality: 0.918803
    Test TPR: 0.531062, Predictive Equality: 0.893073
    
    Model 5 - Params: {'C': 1.5, 'solver': 'newton-cholesky'}
    Training AUC: 0.997665
    Validation TPR: 0.453094, Predictive Equality: 0.694906
    Test TPR: 0.464930, Predictive Equality: 0.560364

    RandomUnderSampler and RandomOverSampler > SMOTESVM >> other re-sampling method like SMOTE and NearMiss
    SMOTESVM is OK (TPR: 0.475) but it's slow and tends to overfit.
    
    Model 5+ - try VAE, GAN, and Diffusion Model to enrich the data, see their results.
    
    '''
    
    '''
    Findings:
    - log-transformation always works, it can improve both performance and fairness score.
    - Newton's methods provide better performance than stochastic optimization methods (e.g. sag, saga)
    - both re-sampling methods and class weight balance techniques can increase the fairness score significantly at the cost of little performance loss.
    - Re-weight > re-sampling
    - random re-sampling method >> SMOTESVM > SMOTE.
    - There's a trade-off between model performance and model fairness, we never see the best-performed model has a high fairness score.
    '''