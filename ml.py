import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml 

# --- Configuration Constants ---

CONFIG = {
    'K': 1024,
    'DATASET_FILE': 'astro_rfi_dataset.npz', 
    'NUM_SAMPLES': 100000,
    'SPLIT_RATIO': 0.9,
    'BATCH_SIZE': 32,
    'LEARNING_RATE': 0.0002,
    'EPOCHS': 10,
    'BETA1': 0.5,
    'BETA2': 0.999,
    'RESULTS_DIR': 'training_results',
    'PLOTS_TO_SAVE': 5,
    'Z_DIM': 16,
    
    # --- NEW CONFIG FOR WINDOWED STD ANALYSIS ---
    'WINDOW_SIZE_STD': 128, #64, # Window size for standard deviation calculation (e.g., 64 samples)
    'STEP_SIZE_STD': 16,   # Step size/overlap for the sliding window
    'EXPECTED_TG_STD': 0.03, # Placeholder for expected clean sky STD (T_g)
}
K = CONFIG['K'] # Convenience constant

# --- MASK STATISTIC FUNCTIONS (KEPT, BUT UNUSED) ---
# Currently not used for plotting in the results
def define_rfi_mask(K, rfi_params, mask_width_factor=3.0):
    """Placeholder for RFI mask definition."""
    t_samples = np.arange(K)
    t0 = rfi_params.get('t0', K/2)
    sigma = rfi_params.get('sigma', 10)
    mask = np.abs(t_samples - t0) <= (mask_width_factor * sigma)
    N_Omega = np.sum(mask)
    if N_Omega > 0:
        min_idx = np.min(np.where(mask)[0])
        max_idx = np.max(np.where(mask)[0])
        window_size_rfi = max_idx - min_idx + 1
    else:
        window_size_rfi = 1 
    return mask, N_Omega, window_size_rfi

def calculate_mask_statistic(signal, mask, N_Omega, T_g_true):
    """Placeholder for Mask Statistic calculation (will return NaN if called without T_g_true)."""
    # This calculation is now disabled by only running the Windowed STD plot
    return np.nan 

# --- WINDOWED STATISTIC FUNCTIONS (USED FOR PLOT 4 in recovery functions) ---
def calculate_window_stats(signal, window_size, step_size):
    """
    Calculates the standard deviation for sliding windows across the signal.
    """
    K = len(signal)
    
    # Ensure starts array doesn't go beyond the signal length minus window size
    starts = np.arange(0, K - window_size + 1, step_size)
    if len(starts) == 0 and K >= window_size: # Handle case where step_size > remaining length
        starts = np.array([0])
    elif len(starts) == 0: # Cannot form a window
        return np.array([]), np.array([])
        
    # Calculate window centers for plotting
    window_centers = starts + window_size / 2 - 0.5
    
    stds = []
    
    for start in starts:
        end = start + window_size
        window = signal[start:end]
        stds.append(np.std(window))
        # stds.append(np.mean(window))
        
    return np.array(window_centers), np.array(stds)



# --- DATA LOADING AND CUSTOM DATASET ---
class AstroRFIDataset(Dataset):
    def __init__(self, inputs, targets):
        # inputs are S' (~S), targets are T_ng (RFI)
        self.inputs = torch.from_numpy(inputs).unsqueeze(1).float() 
        self.targets = torch.from_numpy(targets).unsqueeze(1).float()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def prepare_dataloaders(config, file_path):
    """
    Loads, splits, and prepares data loaders. Returns only essential data.
    """
    if not os.path.exists(file_path):
        print(f"Error: Dataset file '{file_path}' not found.")
        print("Please run a simulation script first to generate the data.")
        # Only return 4 essential None values
        return None, None, None, None
    else:
        data = np.load(file_path, allow_pickle=True)
        inputs = data['inputs']
        targets = data['targets']

    # Split data index
    split_idx = int(config['SPLIT_RATIO'] * len(inputs))
    
    # Split data arrays
    train_inputs, val_inputs = inputs[:split_idx], inputs[split_idx:]
    train_targets, val_targets = targets[:split_idx], targets[split_idx:]
    
    train_dataset = AstroRFIDataset(train_inputs, train_targets)
    val_dataset = AstroRFIDataset(val_inputs, val_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False)
    
    # Return only the 4 essential values: loaders and numpy arrays for plotting
    return train_loader, val_loader, val_inputs, val_targets 

# --- MODEL DEFINITION: RFINet ---
class RFINet(nn.Module):
    """3-layer Encoder/Decoder network using Linear layers for 1D time stream RFI recovery."""
    def __init__(self, input_size, z_dim):
        super(RFINet, self).__init__()
        h1_dim = input_size // 2 
        h2_dim = h1_dim // 2 	
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, h1_dim),
            nn.LeakyReLU(0.02),
            nn.Linear(h1_dim, h2_dim),
            nn.LeakyReLU(0.02),
            nn.Dropout(0.2), 
            nn.Linear(h2_dim, z_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h2_dim),
            nn.LeakyReLU(0.02),
            nn.Linear(h2_dim, h1_dim),
            nn.LeakyReLU(0.02),
            nn.Linear(h1_dim, input_size)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(decoded.size(0), 1, -1)

# --- TRAINER CLASS ---
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
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(self.config['EPOCHS']):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            print(f"Epoch {epoch+1}/{self.config['EPOCHS']}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        print("Training finished.")
        return history

# --- Plotting Training History ---
def plot_training_history(history, config, results_dir):
    """Plots and saves the training and validation loss history."""
    epochs = range(1, config['EPOCHS'] + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], label='Training Loss', marker='o', linestyle='-', markersize=2)
    plt.plot(epochs, history['val_loss'], label='Validation Loss', marker='o', linestyle='-', markersize=2)
    
    plt.title('Training and Validation Loss History', fontsize=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(r'Loss (MSE)', fontsize=12)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.6)
    
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, 'training_history.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"\nTraining history plot saved to {plot_path}")

# --- Plotting Results for RFI Recovery (MODIFIED for Windowed STD Plot) ---
@torch.no_grad()
def plot_rfi_recovery_results(model, config, val_inputs, val_targets, device, results_dir):
    """
    Generates and saves plot showing S, Tng_hat, Tg_hat, and Windowed Standard Deviation.
    """
    model.eval()
    
    num_plots = config['PLOTS_TO_SAVE']
    K = config['K']
    time_samples = np.arange(K)
    
    # Get parameters for Windowed STD
    window_size = config['WINDOW_SIZE_STD']
    step_size = config['STEP_SIZE_STD']
    expected_std = config['EXPECTED_TG_STD']

    if len(val_inputs) < num_plots:
        num_plots = len(val_inputs)
        if num_plots == 0:
            print("No validation data to plot. Skipping results plot.")
            return

    # Data preparation
    inputs_tensor = torch.from_numpy(val_inputs[:num_plots]).unsqueeze(1).to(device).float()
    recovered_rfi_tensor = model(inputs_tensor)
    
    recovered_rfi = recovered_rfi_tensor.cpu().numpy().squeeze()
    true_rfi = val_targets[:num_plots]
    input_signals = val_inputs[:num_plots]

    # Plot Setup
    # 4 columns for S, Tng vs Tng_hat, Tg_hat, and Windowed STD
    fig, axes = plt.subplots(num_plots, 4, figsize=(20, 3 * num_plots))
    if num_plots == 1:
        axes = np.expand_dims(axes, axis=0)
        
    os.makedirs(results_dir, exist_ok=True)
    
    # Loop for Each Example
    for i in range(num_plots):
        input_signal = input_signals[i]
        recovered_rfi_signal = recovered_rfi[i]
        # Calculate the CLEAN SKY SIGNAL as the residual: T_g_hat = S - T_ng_hat
        clean_sky_signal = input_signal - recovered_rfi_signal
        
        # --- Windowed Standard Deviation Calculation ---
        S_centers, S_stds = calculate_window_stats(input_signal, window_size, step_size)
        clean_centers, clean_stds = calculate_window_stats(clean_sky_signal, window_size, step_size)

        # Plotting
        max_val = np.max(np.abs(input_signal)) * 1.1
        
        # COL 1: Input Signal (S)
        axes[i, 0].plot(time_samples, input_signal, color='red', alpha=0.8)
        axes[i, 0].set_title(f'Ex {i+1}: Input Signal (S)')
        axes[i, 0].set_ylabel('Amplitude')
        axes[i, 0].set_ylim(-max_val, max_val)

        # COL 2: True RFI vs. Recovered RFI 
        axes[i, 1].plot(time_samples, true_rfi[i], label=r'True RFI ($T_{{ng}}$)', color='orange')
        axes[i, 1].plot(time_samples, recovered_rfi_signal, label=r'Recovered RFI ($\hat{{T}}_{{ng}}$)', color='purple')
        axes[i, 1].set_title(r'RFI Recovery ($T_{{ng}}$ vs $\hat{{T}}_{{ng}}$)')
        axes[i, 1].legend(loc='lower left', fontsize=8)
        axes[i, 1].set_ylabel('Amplitude')
        axes[i, 1].set_ylim(-max_val, max_val)

        # COL 3: Clean Sky Signal 
        axes[i, 2].plot(time_samples, input_signal, color='red', alpha=0.3, label='Original Input (S)')
        axes[i, 2].plot(time_samples, clean_sky_signal, label=r'Clean Sky Residual ($\hat{{T}}_g=S-\hat{{T}}_{{ng}}$)', color='blue')
        axes[i, 2].set_title(r'Clean Sky Signal ($\hat{{T}}_g$)')
        axes[i, 2].legend(loc='lower left', fontsize=8)
        axes[i, 2].set_ylabel('Amplitude')
        axes[i, 2].set_ylim(-max_val, max_val)
        
        # COL 4: Windowed Standard Deviation Comparison (NEW PLOT)
        # axes[i, 3].plot(S_centers, S_stds, label=r'Input $\sigma$', color='red', linestyle='--')
        # axes[i, 3].plot(clean_centers, clean_stds, label=r'Clean $\hat{T}_g \sigma$', color='blue')
        axes[i, 3].scatter(S_centers, S_stds, label=r'Input $\sigma$', color='red')
        axes[i, 3].scatter(clean_centers, clean_stds, label=r'Clean $\hat{T}_g \sigma$', color='blue')
        # axes[i, 3].axhline(expected_std, color='k', linestyle=':', linewidth=1, label=r'Expected $\sigma(T_g)$')
        
        axes[i, 3].set_title(f'$\sigma$ (Win Size={window_size})')
        axes[i, 3].set_xlabel('Window Center Sample Number')
        axes[i, 3].set_ylabel(f'$\sigma$')
        axes[i, 3].legend(loc='upper right', fontsize=8)
        axes[i, 3].grid(axis='y', alpha=0.5)

        # Set common formatting
        for ax in axes[i, :]:
            if i == num_plots - 1:
                ax.set_xlabel('Sample Number' if ax != axes[i, 3] else 'Window Center')
            ax.grid(True, alpha=0.3)
            
    plt.suptitle("Supervised RFI Recovery and Windowed Standard Deviation Analysis", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plot_path = os.path.join(results_dir, 'rfi_recovery_and_windowed_std.png')
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"\nResults plot saved to {plot_path}")

# --- Save Residual and predicted RFI ---
@torch.no_grad()
def save_test_predictions(model, val_loader, device, results_dir, filename='test_results.npz'):
    """
    Runs inference on the test set and saves:
    1. The RFI Estimation Error (True RFI - Predicted RFI)
    2. The Predicted RFI (Model Output)
    3. The Original Input (for context)
    """
    model.eval()
    all_inputs = []
    all_targets = []
    all_predictions = []

    print(f"Generating predictions for the test set...")
    for inputs, targets in val_loader:
        inputs_dev = inputs.to(device)
        outputs = model(inputs_dev)
        
        all_inputs.append(inputs.squeeze().numpy())
        all_targets.append(targets.squeeze().numpy())
        all_predictions.append(outputs.cpu().squeeze().numpy())

    # Concatenate lists into single numpy arrays
    inputs_np = np.concatenate(all_inputs, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)
    predictions_np = np.concatenate(all_predictions, axis=0)

    # Calculate the RFI Error (Residual RFI after model subtraction)
    # This represents how much of the RFI the model missed or over-estimated
    rfi_error_np = targets_np - predictions_np

    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, filename)
    
    np.savez(
        save_path, 
        inputs=inputs_np,             # Original S'
        rfi_error=rfi_error_np,       # (T_ng - T_ng_hat)
        predictions=predictions_np    # T_ng_hat
    )
    
    print(f"Test results saved to {save_path}")
    print(f"Shape of saved data: {inputs_np.shape}")


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Prepare DataLoaders
    data_file_path = CONFIG['DATASET_FILE']
    print(f"Loading data using {data_file_path}...")
    
    # Returns 4 values: loaders and raw numpy arrays for validation
    train_loader, val_loader, val_inputs, val_targets = prepare_dataloaders(CONFIG, data_file_path)
    
    if train_loader is None:
        print("\nData loading failed. Please ensure the dataset file exists.")
    else:
        # 3. Initialize Model and Trainer
        model = RFINet(input_size=CONFIG['K'], z_dim=CONFIG['Z_DIM']).to(device)
        trainer = Trainer(model, CONFIG, device)

        # 4. Run Training
        training_history = trainer.run(train_loader, val_loader)

        # 5. Plot Training History (Loss Curves)
        plot_training_history(training_history, CONFIG, CONFIG['RESULTS_DIR'])

        # 6. Plot RFI Recovery for visual inspection (S, Tng, Tg_hat, and STD)
        plot_rfi_recovery_results(
            model, CONFIG, val_inputs, val_targets, device, 
            CONFIG['RESULTS_DIR']
        )
        
        # 7. NEW: Write out the test dataset with RFI Errors and Predictions
        # This saves: inputs, (targets - predictions), and predictions
        test_results_filename = 'astro_rfi_test_residuals.npz'
        save_test_predictions(
            model, 
            val_loader, 
            device, 
            CONFIG['RESULTS_DIR'], 
            filename=test_results_filename
        )
        
        print(f"\n" + "="*50)
        print("PIPELINE COMPLETE")
        print(f"Training Results Directory: {CONFIG['RESULTS_DIR']}")
        print(f"Test Dataset saved as: {test_results_filename}")
        print(f"Keys saved: ['inputs', 'rfi_error', 'predictions']")
        print("="*50)