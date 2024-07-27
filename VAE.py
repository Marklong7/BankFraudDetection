import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Load only the training data
X_train = pd.read_csv("data/X_train_oh_1.csv")
y_train = pd.read_csv("data/y_train_1.csv").iloc[:, 0]

# Identify the minority class
minority_class = y_train.value_counts().idxmin()

# Filter the minority class samples
X_train_minority = X_train[y_train == minority_class]
y_train_minority = y_train[y_train == minority_class]

# Split into train and validation sets
X_train_min, X_val_min, y_train_min, y_val_min = train_test_split(X_train_minority, y_train_minority, test_size=0.2, random_state=42)

# Preprocess the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_min)
X_val_scaled = scaler.transform(X_val_min)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
X_val_tensor = torch.FloatTensor(X_val_scaled)

# Create DataLoaders
train_dataset = TensorDataset(X_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_dataset = TensorDataset(X_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

def train_vae(model, train_loader, val_loader, optimizer, num_epochs, early_stop_patience):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            x = batch[0]
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = loss_function(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0]
                recon_x, mu, logvar = model(x)
                loss = loss_function(recon_x, x, mu, logvar)
                val_loss += loss.item()
        
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        
        #print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve == early_stop_patience:
            print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(f'Early stopping triggered after epoch {epoch+1}')
            break
    
    return best_val_loss

# Hyperparameter grid
param_grid = {
    'hidden_dim': [64, 100, 128, 200, 256, 300],
    'latent_dim': [10, 15, 20, 25],
    'learning_rate': [5e-4, 1e-3, 2e-3]
}

best_val_loss = float('inf')
best_params = None
best_model = None

for hidden_dim in param_grid['hidden_dim']:
    for latent_dim in param_grid['latent_dim']:
        for lr in param_grid['learning_rate']:
            print(f"Training with hidden_dim={hidden_dim}, latent_dim={latent_dim}, lr={lr}")
            
            model = VAE(input_dim=X_train.shape[1], hidden_dim=hidden_dim, latent_dim=latent_dim)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            
            val_loss = train_vae(model, train_loader, val_loader, optimizer, num_epochs=500, early_stop_patience=10)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = {'hidden_dim': hidden_dim, 'latent_dim': latent_dim, 'lr': lr}
                best_model = model

print(f"Best parameters: {best_params}")
print(f"Best validation loss: {best_val_loss:.4f}")

# Generate new samples using the best model
best_model.eval()
with torch.no_grad():
    num_samples_to_generate = len(y_train[y_train != minority_class]) - len(y_train[y_train == minority_class])
    z = torch.randn(num_samples_to_generate, best_params['latent_dim'])
    generated_samples = best_model.decode(z).numpy()

# Inverse transform the generated samples
generated_samples = scaler.inverse_transform(generated_samples)

# Create DataFrames for the generated data
X_train_1_VAE = pd.DataFrame(generated_samples, columns=X_train.columns)
y_train_1_VAE = pd.Series([minority_class] * num_samples_to_generate)

# Save the generated data to new CSV files
X_train_1_VAE.to_csv("X_train_3_VAE.csv", index=False)
y_train_1_VAE.to_csv("y_train_3_VAE.csv", index=False)

print(f"Generated {num_samples_to_generate} new samples for the minority class.")
print("Generated data saved to X_train_3_VAE.csv and y_train_3_VAE.csv")

'''
Best parameters: {'hidden_dim': 200, 'latent_dim': 20, 'lr': 0.002}
Best validation loss: 27.4279
Generated 259516 new samples for the minority class.
Generated data saved to X_train_1_VAE.csv and y_train_1_VAE.csv

Best parameters: {'hidden_dim': 256, 'latent_dim': 20, 'lr': 0.002}
Best validation loss: 26.8644
Generated 259516 new samples for the minority class.
Generated data saved to X_train_2_VAE.csv and y_train_2_VAE.csv
'''