import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SVMSMOTE
from typing import Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.device_count() > 1:
    print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")

# Define the PyTorch model with dropout
class FraudDetectionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(FraudDetectionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

def evaluate(predictions: np.ndarray, ground_truth: np.ndarray, X: np.ndarray, fixed_fpr: float = 0.05) -> list[float]:
    fprs, tprs, thresholds = roc_curve(ground_truth, predictions)
    
    tpr = tprs[fprs < fixed_fpr][-1]
    fpr = fprs[fprs < fixed_fpr][-1]
    threshold = thresholds[fprs < fixed_fpr][-1]
    
    # Assuming customer_age is the last column in X
    sorted_ages = np.sort(X[:, -1])
    young_threshold = sorted_ages[int(0.95 * len(sorted_ages))]
    
    groups = (X[:, -1] > young_threshold).astype(str)
    groups[groups == 'True'] = '>young_threshold'
    groups[groups == 'False'] = '<=young_threshold'
    
    predictive_equality, _ = get_fairness_metrics(ground_truth, predictions, groups, fixed_fpr)
    
    return [round(tpr, 10), round(predictive_equality, 6)]

# Function to get fairness metrics
def get_fairness_metrics(y_true: np.ndarray, y_pred: np.ndarray, groups: pd.Series, fixed_fpr: float = 0.05) -> tuple[float, pd.DataFrame]:
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

def train_val_test(X_train_scaled: np.ndarray, y_train: np.ndarray, X_val_scaled: np.ndarray, y_val: np.ndarray,
                   X_test_scaled: np.ndarray, y_test: np.ndarray, 
                   input_dim: int, hidden_dim: int, output_dim: int, epochs: int, batch_size: int,
                   learning_rate: float, early_stop: int, dropout_rate: float):

    # Convert to PyTorch tensors
    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32),
                                torch.tensor(y_val, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32),
                                 torch.tensor(y_test, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model, loss function, and optimizer
    model = FraudDetectionModel(input_dim, hidden_dim, output_dim, dropout_rate)
    model.to(device)  # Move model to device
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Early stopping variables
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to device
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        # Early stopping check
        if val_loss < best_val_loss:
            print(f"New best model found at epoch {epoch+1}")
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model_resample.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve == early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_model_resample.pth'))
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_preds = []
        val_labels = []
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            val_preds.append(outputs.cpu().numpy())
            val_labels.append(labels.cpu().numpy())
    
    val_preds = np.concatenate(val_preds)
    val_labels = np.concatenate(val_labels)
    
    # Calculate metrics for validation
    val_auc_score = roc_auc_score(val_labels, val_preds)
    val_tpr = evaluate(val_preds, val_labels, X_val_scaled)[0]
    val_equality = evaluate(val_preds, val_labels, X_val_scaled)[1]
    
    # Testing
    with torch.no_grad():
        test_preds = []
        test_labels = []
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            test_preds.append(outputs.cpu().numpy())
            test_labels.append(labels.cpu().numpy())
    
    test_preds = np.concatenate(test_preds)
    test_labels = np.concatenate(test_labels)
    
    # Calculate metrics for testing
    test_auc_score = roc_auc_score(test_labels, test_preds)
    test_tpr = evaluate(test_preds, test_labels, X_test_scaled)[0]
    test_equality = evaluate(test_preds, test_labels, X_test_scaled)[1]
    
    return val_auc_score, val_tpr, val_equality, test_auc_score, test_tpr, test_equality

def read_data():
    # Load data from CSV files
    X_train = pd.read_csv("data/X_train_oh_1.csv")
    X_val = pd.read_csv("data/X_val_oh_1.csv")
    X_test = pd.read_csv("data/X_test_oh_1.csv")
    y_train = pd.read_csv("data/y_train_1.csv").values.ravel()
    y_val = pd.read_csv("data/y_val_1.csv").values.ravel()
    y_test = pd.read_csv("data/y_test_1.csv").values.ravel()

    # Return data as a dictionary
    data = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
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

# Main function
def main():
    # Read data
    data = read_data()
    X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
    y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']
    
    # Apply resampling technique if specified
    resample = 'SVMSMOTE'  # Set this to 'SMOTE', 'RandomOverSampler', 'RandomUnderSampler', or 'SVMSMOTE' to apply resampling
    if resample:
        X_train, y_train = apply_sampling_technique(X_train, y_train, resample)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    print("Scaling is complete.")
    
    # Define hyperparameters
    hyperparameters = {
        'input_dim': X_train_scaled.shape[1],
        'hidden_dim': 200,
        'output_dim': 1,
        'epochs': 50,
        'batch_size': 2048,
        'learning_rate': 0.0005,
        'early_stop': 10,
        'dropout_rate': 0.5
    }

    # Call the function with training, validation, and testing data
    val_auc_score, val_tpr, val_equality, test_auc_score, test_tpr, test_equality = train_val_test(
        X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, **hyperparameters
    )

    # Print hyperparameters
    print("Hyperparameters:")
    for key, value in hyperparameters.items():
        print(f"  {key}: {value}")
    print(f"  resample: {resample}")
    print("----------------------------------------------")

    # Print performance metrics
    print("Validation Results:")
    print(f"  AUC Score: {val_auc_score:.4f}")
    print(f"  True Positive Rate (TPR): {val_tpr:.4f}")
    print(f"  Predictive Equality: {val_equality:.4f}")
    print("----------------------------------------------")
    print("Testing Results:")
    print(f"  AUC Score: {test_auc_score:.4f}")
    print(f"  True Positive Rate (TPR): {test_tpr:.4f}")
    print(f"  Predictive Equality: {test_equality:.4f}")

if __name__ == "__main__":
    main()
    '''
    Hyperparameters:
  input_dim: 50
  hidden_dim: 200
  output_dim: 1
  epochs: 50
  batch_size: 2048
  learning_rate: 0.0004
  early_stop: 10
  dropout_rate: 0.5
  resample: SMOTE
----------------------------------------------
Validation Results:
  AUC Score: 0.8798
  True Positive Rate (TPR): 0.4950
  Predictive Equality: 0.7382
----------------------------------------------
Testing Results:
  AUC Score: 0.8819
  True Positive Rate (TPR): 0.5210
  Predictive Equality: 0.5340
  '''
    
'''
Hyperparameters:                                                                                                           
  input_dim: 50
  hidden_dim: 200
  output_dim: 1
  epochs: 50
  batch_size: 2048
  learning_rate: 0.0004
  early_stop: 10
  dropout_rate: 0.5
  resample: RandomOverSampler
----------------------------------------------
Validation Results:
  AUC Score: 0.8730
  True Positive Rate (TPR): 0.4691
  Predictive Equality: 0.8350
----------------------------------------------
Testing Results:
  AUC Score: 0.8740
  True Positive Rate (TPR): 0.4990
  Predictive Equality: 0.5896


Hyperparameters:
  input_dim: 50
  hidden_dim: 200
  output_dim: 1
  epochs: 50
  batch_size: 2048
  learning_rate: 0.0005
  early_stop: 10
  dropout_rate: 0.5
  resample: RandomUnderSampler
----------------------------------------------
Validation Results:
  AUC Score: 0.8695
  True Positive Rate (TPR): 0.4691
  Predictive Equality: 0.9298
----------------------------------------------
Testing Results:
  AUC Score: 0.8801
  True Positive Rate (TPR): 0.4950
  Predictive Equality: 0.8941
  '''