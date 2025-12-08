import numpy as np
import os

# --- Configuration Constants ---
K = 1024    # Number of samples (time stream length)
k0 = 256    # Characteristic frequency for the sky signal power spectrum
beta = 0.0002
a = 1.0
NUM_SAMPLES = 100000 
DATASET_FILE = 'astro_rfi_dataset.npz'

# --- Signal Generation Functions ---
def noise_power_spectrum(K, k0, beta, a):
    """Generates the 1D power spectrum P(k) for the sky signal (Equation 5)."""
    P_k = np.zeros(K)
    k_array = np.arange(K // 2 + 1)
    
    # P(k) = a * (1 + exp(-beta * (k - k0)^2)) * exp(-k / k0)
    P_k_half = a * (1 + np.exp(-beta * (k_array - k0)**2)) * np.exp(-k_array / k0)
    
    P_k[0 : K // 2 + 1] = P_k_half
    P_k[K // 2 + 1 :] = P_k_half[K // 2 - 1 : 0 : -1]
    return P_k

def generate_sky_signal(K, k0, beta, a):
    """Generates the Gaussian random sky signal T_g."""
    # Generate complex Gaussian variates X_k
    X_k_real = np.random.normal(0, 1, size=K // 2 + 1)
    X_k_imag = np.random.normal(0, 1, size=K // 2 + 1)
    X_k = X_k_real + 1j * X_k_imag
    
    # Enforce real-valued time stream properties
    X_k[0] = X_k[0].real
    if K % 2 == 0:
        X_k[-1] = X_k[-1].real
        
    P_k = noise_power_spectrum(K, k0, beta, a)
    T_g_freq = X_k * np.sqrt(P_k[0 : K // 2 + 1])
    
    full_T_g_freq = np.zeros(K, dtype=complex)
    full_T_g_freq[0 : K // 2 + 1] = T_g_freq
    full_T_g_freq[K // 2 + 1 :] = np.conj(T_g_freq[1:K // 2])[::-1]
    
    T_g = np.fft.ifft(full_T_g_freq).real
    
    # Normalization
    return T_g / np.sqrt(np.mean(T_g**2)) * 0.03

def generate_rfi_signal(K):
    """Generates the non-Gaussian RFI signal T_ng (Equation 7)."""
    t_samples = np.arange(K)
    
    # Randomly select RFI parameters
    A0 = np.random.uniform(0.10, 0.20)
    phi = np.random.uniform(0, 2 * np.pi)
    # nu = np.random.uniform(0.2, 0.7)
    # nu = np.random.uniform(0.3, 0.6)
    nu = np.random.uniform(0.05, 0.25)
    t0 = np.random.uniform(300, 700)
    # sigma = np.random.uniform(50, 150)
    sigma = np.random.uniform(20, 50)
    
    # T_ng = A0 * cos(phi + 2 * pi * nu * t) * exp(-(t - t0)^2 / (2 * sigma^2))
    T_ng = A0 * np.cos(phi + 2 * np.pi * nu * t_samples) * np.exp(-(t_samples - t0)**2 / (2 * sigma**2))
    return T_ng

def generate_training_data(num_samples, K, k0, beta, a):
    """
    Generates training data for RFI RECOVERY.
    Inputs (S) = T_g + T_ng
    Targets (T_ng) = RFI signal.
    """
    inputs = np.zeros((num_samples, K), dtype=np.float32)
    targets = np.zeros((num_samples, K), dtype=np.float32)

    for i in range(num_samples):
        T_g = generate_sky_signal(K, k0, beta, a)
        T_ng = generate_rfi_signal(K)
        
        # S = T_g + T_ng (Combined Signal with RFI)
        S = T_g + T_ng
        
        inputs[i] = S
        targets[i] = T_ng  # TARGET IS THE RFI SIGNAL (T_ng)
        
    return inputs, targets


if __name__ == "__main__":
    np.random.seed(42) # Set seed for reproducibility
    print(f"Generating dataset of {NUM_SAMPLES} time streams...")
    
    inputs, targets = generate_training_data(NUM_SAMPLES, K, k0, beta, a)
    
    # Save the dataset
    np.savez_compressed(DATASET_FILE, inputs=inputs, targets=targets)
    print(f"Dataset saved successfully to {DATASET_FILE}")
    print(f"Input shape (S=Tg+Tng): {inputs.shape}")
    print(f"Target shape (Tng): {targets.shape}")
