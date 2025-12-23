import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import json
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 30
L2_REG = 1e-4  # L2 Regularization
SEED = 42
SENSOR_NAME = "CFT24"
MODEL_NAME = f"{SENSOR_NAME}_MLP"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CONFIG_DIR = os.path.join(SCRIPT_DIR, '..', 'hardware_configs')

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seed for reproducibility 
torch.manual_seed(SEED)
np.random.seed(SEED)

# --- 1. DATASET CLASS ---
class CoinFTDataset(Dataset):
    def __init__(self, h5_path):
        # Load data into memory 
        with h5py.File(h5_path, 'r') as f:
            self.X = torch.tensor(f['data'][:], dtype=torch.float32)
            self.Y = torch.tensor(f['label'][:], dtype=torch.float32)
            
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# --- 2. NETWORK ARCHITECTURE ---
class CoinFTNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CoinFTNet, self).__init__()
        self.net = nn.Sequential(
            
            # Layer 1: 128
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            
            # Layer 2: 64
            nn.Linear(128, 64),
            nn.ReLU(),
            
            # Layer 3: 36
            nn.Linear(64, 36),
            nn.ReLU(),
            
            # Layer 4: 24
            nn.Linear(36, 24),
            nn.ReLU(),
            
            # Layer 5: 12
            nn.Linear(24, 12),
            nn.ReLU(),
            
            # Output Layer: 6 (Regression)
            nn.Linear(12, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def train_pipeline():
    # --- 3. DATA LOADING ---
    print("Loading datasets...")
    train_dataset = CoinFTDataset(os.path.join(DATA_DIR, 'train.h5'))
    val_dataset   = CoinFTDataset(os.path.join(DATA_DIR, 'val.h5'))
    test_dataset  = CoinFTDataset(os.path.join(DATA_DIR, 'test.h5'))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # Test loader: batch_size = total size to predict all at once later
    test_loader  = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    # Determine dimensions dynamically
    input_dim = train_dataset.X.shape[1]  # Should be 12 based on CoinFT raw data
    output_dim = train_dataset.Y.shape[1] # Should be 6 (Fx...Mz)
    print(f"Input Dim: {input_dim}, Output Dim: {output_dim}")

    # --- 4. MODEL SETUP ---
    model = CoinFTNet(input_dim, output_dim).to(device)
    
    # Loss function (RegressionLayer)
    criterion = nn.MSELoss()
    
    # Optimizer (Adam with L2 Regularization)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG)

    # --- 5. TRAINING LOOP ---
    print("\nStarting training...")
    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            
            optimizer.zero_grad()           # Clear gradients
            outputs = model(X_batch)        # Forward pass
            loss = criterion(outputs, Y_batch) # Compute loss
            loss.backward()                 # Backward pass
            optimizer.step()                # Update weights
            
            running_loss += loss.item() * X_batch.size(0)

        epoch_train_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_train_loss)

        # Validation Step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, Y_val in val_loader:
                X_val, Y_val = X_val.to(device), Y_val.to(device)
                preds = model(X_val)
                loss = criterion(preds, Y_val)
                val_loss += loss.item() * X_val.size(0)
        
        epoch_val_loss = val_loss / len(val_dataset)
        val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")

    print("Training complete.")

    # --- 6. SAVE MODEL (PyTorch & ONNX) ---
    pth_path = os.path.join(CONFIG_DIR, f"{MODEL_NAME}.pth")
    onnx_path = os.path.join(CONFIG_DIR, f"{MODEL_NAME}.onnx")

    # Save standard PyTorch model
    torch.save(model.state_dict(), pth_path)
    
    # Export to ONNX
    dummy_input = torch.randn(1, input_dim).to(device)
    torch.onnx.export(model, dummy_input, onnx_path, 
                      input_names=['input'], output_names=['output'])
    
    print(f"Models saved to:\n  {pth_path}\n  {onnx_path}")

    # --- 7. EVALUATION & DENORMALIZATION ---
    print("\nEvaluating on Test Set...")
    
    # Load Normalization Constants
    with open(os.path.join(CONFIG_DIR, f'{SENSOR_NAME}_norm.json'), 'r') as f:
        norm = json.load(f)
        mu_y = np.array(norm['mu_y'])
        sd_y = np.array(norm['sd_y'])

    model.eval()
    with torch.no_grad():
        # Get all test data
        X_test, Y_test_norm = next(iter(test_loader))
        X_test = X_test.to(device)
        
        # Predict (Normalized Space)
        pred_norm = model(X_test).cpu().numpy()
        Y_test_norm = Y_test_norm.numpy()

        # Denormalize: pred = pred_n * std + mean
        pred_real = (pred_norm * sd_y) + mu_y
        Y_test_real = (Y_test_norm * sd_y) + mu_y

    # Calculate MSE per axis
    mse_per_dim = np.mean((pred_real - Y_test_real)**2, axis=0)
    
    labels = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
    print("\nMean-squared error per axis (Newtons / Nm):")
    for i, label in enumerate(labels):
        print(f"  {label} : {mse_per_dim[i]:.6f}")

    # --- 8. VISUAL CHECK (Matplotlib) ---
    plot_len = len(Y_test_real)
    
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    axs = axs.ravel()
    
    for i in range(6):
        axs[i].plot(Y_test_real[:plot_len, i], 'b-', label='True')
        axs[i].plot(pred_real[:plot_len, i], 'r--', label='Pred')
        axs[i].set_title(f'Axis {labels[i]}')
        if i == 0: axs[i].legend()
        
    plt.tight_layout()

    fig_path = os.path.join(DATA_DIR, f"{MODEL_NAME}_results.png")
    plt.savefig(fig_path)
    plt.show()

if __name__ == "__main__":
    train_pipeline()