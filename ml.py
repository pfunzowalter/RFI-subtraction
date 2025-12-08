import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml

# --- Configuration Constants (Unchanged) ---

CONFIG = {
    'K': 1024,
    'DATASET_FILE': 'astro_rfi_dataset.npz',
    'NUM_SAMPLES': 100000,
    'SPLIT_RATIO': 0.9,
    'BATCH_SIZE': 32,
    'LEARNING_RATE': 0.0002,
    'EPOCHS': 100,
    'BETA1': 0.5,
    'BETA2': 0.999,
    'RESULTS_DIR': 'training_results',
    'PLOTS_TO_SAVE': 5,
    'Z_DIM': 16,
}
K = CONFIG['K'] # Convenience constant

# --- Data Loading and Custom Dataset ---
class AstroRFIDataset(Dataset):
    def __init__(self, inputs, targets):
        # inputs are S (S=Tg+Tng), targets are T_ng (RFI)
        # Add channel dimension (1 channel, K samples)
        self.inputs = torch.from_numpy(inputs).unsqueeze(1) 
        self.targets = torch.from_numpy(targets).unsqueeze(1)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def prepare_dataloaders(config, file_path):
    """Loads, splits, and prepares data loaders from the NPZ file."""
    
    if not os.path.exists(file_path):
        print(f"Error: Dataset file '{file_path}' not found.")
        print("Please run 'python signalsimulation.py' first to generate the data.")
        return None, None, None, None
    else:
        # Load data where Target is T_ng (RFI)
        data = np.load(file_path)
        inputs = data['inputs']
        targets = data['targets']

    # Split data
    split_idx = int(config['SPLIT_RATIO'] * len(inputs))
    train_inputs, val_inputs = inputs[:split_idx], inputs[split_idx:]
    train_targets, val_targets = targets[:split_idx], targets[split_idx:]
    
    train_dataset = AstroRFIDataset(train_inputs, train_targets)
    val_dataset = AstroRFIDataset(val_inputs, val_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False)
    
    return train_loader, val_loader, val_inputs, val_targets

# --- Model Definition: RFINet ---
class RFINet(nn.Module):
    """3-layer Encoder/Decoder network using Linear layers for 1D time stream RFI recovery."""
    def __init__(self, input_size, z_dim):
        super(RFINet, self).__init__()
        
        h1_dim = input_size // 2 # 512
        h2_dim = h1_dim // 2 	# 256
        
        # --- Encoder (Compression) ---
        self.encoder = nn.Sequential(
            nn.Linear(input_size, h1_dim),
            nn.LeakyReLU(0.02),
            nn.Linear(h1_dim, h2_dim),
            nn.LeakyReLU(0.02),
            nn.Dropout(0.2), 
            nn.Linear(h2_dim, z_dim) # Bottleneck (16 samples)
        )
        
        # --- Decoder (Reconstruction) ---
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h2_dim),
            nn.LeakyReLU(0.02),
            nn.Linear(h2_dim, h1_dim),
            nn.LeakyReLU(0.02),
            nn.Linear(h1_dim, input_size) # Output (1024 samples)
        )

    def forward(self, x):
        # Flatten input from (batch, 1, K) to (batch, K)
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # Reshape output back to (batch, 1, K)
        return decoded.view(decoded.size(0), 1, -1)

# --- Trainer Class ---
class Trainer:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=config['LEARNING_RATE'], 
            betas=(config['BETA1'], config['BETA2'])
        )

    def train_epoch(self, data_loader):
        self.model.train()
        running_loss = 0.0
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss / len(data_loader)

    @torch.no_grad()
    def validate(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            total_loss += loss.item()
        self.model.train()
        return total_loss / len(data_loader)
    
    def run(self, train_loader, val_loader):
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters() if p.requires_grad)} trainable parameters.")
        print("Starting training...")
        
        # Initialize lists to store training history
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(self.config['EPOCHS']):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            # Store the losses
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            print(f"Epoch {epoch+1}/{self.config['EPOCHS']}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        print("Training finished.")
        # Return the history dictionary
        return history

# --- Plotting Function for Training History ---
def plot_training_history(history, config, results_dir):
    """Plots and saves the training and validation loss history."""
    
    epochs = range(1, config['EPOCHS'] + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], label='Training Loss', marker='o', linestyle='-', markersize=2)
    plt.plot(epochs, history['val_loss'], label='Validation Loss', marker='o', linestyle='-', markersize=2)
    
    plt.title('Training and Validation Loss History', fontsize=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(r'Loss (MSE)', fontsize=12)
    plt.yscale('log') # Log scale is often better for loss plots
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.6)
    
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, 'training_history.png')
    plt.savefig(plot_path)
    plt.close() 
    print(f"\nTraining history plot saved to {plot_path}")

# --- Plotting Results for RFI Recovery ---
@torch.no_grad()
def plot_rfi_recovery_results(model, config, val_inputs, val_targets, device, results_dir):
    """Generates and saves plot showing true RFI, recovered RFI, and the final clean sky residual."""
    model.eval()
    
    # Prepare data for plotting
    inputs_tensor = torch.from_numpy(val_inputs[:config['PLOTS_TO_SAVE']]).unsqueeze(1).to(device)
    
    # Get network output (recovered RFI: T_ng_hat)
    recovered_rfi_tensor = model(inputs_tensor)
    
    # Convert back to numpy
    recovered_rfi = recovered_rfi_tensor.cpu().numpy().squeeze()
    true_rfi = val_targets[:config['PLOTS_TO_SAVE']]
    input_signals = val_inputs[:config['PLOTS_TO_SAVE']]
    
    num_plots = config['PLOTS_TO_SAVE']
    K = config['K']
    time_samples = np.arange(K)

    fig, axes = plt.subplots(num_plots, 3, figsize=(16, 3 * num_plots))
    os.makedirs(results_dir, exist_ok=True)
    
    for i in range(num_plots):
        input_signal = input_signals[i]
        true_rfi_signal = true_rfi[i]
        recovered_rfi_signal = recovered_rfi[i]
        
        # Calculate the CLEAN SKY SIGNAL as the residual: T_g_hat = S - T_ng_hat
        clean_sky_signal = input_signal - recovered_rfi_signal 

        max_val = np.max(np.abs(input_signal)) * 1.1
        
        # 1. Input Signal (S = T_g + T_ng)
        axes[i, 0].plot(time_samples, input_signal, color='red', alpha=0.8)
        axes[i, 0].set_title(f'Ex {i+1}: Input Signal (S)')
        axes[i, 0].set_ylabel('Amplitude')
        axes[i, 0].set_ylim(-max_val, max_val)

        # 2. True RFI vs. Recovered RFI (Network Target)
        axes[i, 1].plot(time_samples, true_rfi_signal, label=r'True RFI ($T_{{ng}}$)', color='orange')
        axes[i, 1].plot(time_samples, recovered_rfi_signal, label=r'Recovered RFI ($\hat{{T}}_{{ng}}$)', color='purple') #, linestyle='--')
        axes[i, 1].set_title(r'RFI Recovery ($T_{{ng}}$ vs $\hat{{T}}_{{ng}}$)')
        axes[i, 1].legend(loc='lower left', fontsize=8)
        axes[i, 1].set_ylabel('Amplitude')
        axes[i, 1].set_ylim(-max_val, max_val)

        # 3. Clean Sky Signal (The goal result)
        axes[i, 2].plot(time_samples, input_signal, color='red', alpha=0.3, label='Original Input (S)')
        axes[i, 2].plot(time_samples, clean_sky_signal, label=r'Clean Sky Residual ($\hat{{T}}_g=S-\hat{{T}}_{{ng}}$)', color='blue')
        axes[i, 2].set_title(r'Clean Sky Signal ($\hat{{T}}_g$)')
        axes[i, 2].legend(loc='lower left', fontsize=8)
        axes[i, 2].set_ylabel('Amplitude')
        axes[i, 2].set_ylim(-max_val, max_val)
        
        for ax in axes[i, :]:
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Sample Number')
            
    plt.suptitle("Supervised RFI Recovery: Isolating the Non-Gaussian Component", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plot_path = os.path.join(results_dir, 'rfi_recovery_examples.png')
    plt.savefig(plot_path)
    plt.close(fig) # Close the figure to free memory
    print(f"\nResults plot saved to {plot_path}")


# --- Main Execution (Modified) ---
if __name__ == "__main__":
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Prepare DataLoaders
    data_file_path = CONFIG['DATASET_FILE']
    print(f"Loading data using {data_file_path}...")
    
    train_loader, val_loader, val_inputs, val_targets = prepare_dataloaders(CONFIG, data_file_path)
    
    if train_loader is None:
        print("\nData loading failed. Please ensure signalsimulation.py has been run successfully to create the dataset.")
    else:
        # 3. Initialize Model and Trainer
        model = RFINet(input_size=CONFIG['K'], z_dim=CONFIG['Z_DIM']).to(device)
        trainer = Trainer(model, CONFIG, device)

        # 4. Run Training and capture history
        training_history = trainer.run(train_loader, val_loader)

        # 5. Plot Training History
        plot_training_history(training_history, CONFIG, CONFIG['RESULTS_DIR'])

        # 6. Plot and Save Results
        plot_rfi_recovery_results(model, CONFIG, val_inputs, val_targets, device, CONFIG['RESULTS_DIR'])
        
        print("\nTraining complete. Check the 'training_results' directory for the plots.")
