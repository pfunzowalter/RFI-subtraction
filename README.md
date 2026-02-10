# RFI Subtraction with DEEP LEARNING
Useful Resources: https://iopscience.iop.org/article/10.1088/1538-3873/ac98e1/pdf

The current project is configured for Supervised RFI Recovery (We want the network to output the RFI component), the data structure is built on the fundamental relationship:$$\text{Observed Signal} (S) = \text{Clean Sky Signal} (T_g) + \text{RFI Signal} (T_{ng})$$
Data Structure for training process is supervised: we provide the model with an Input and the corresponding desired Target. Input (X)$S$ (Observed Signal) which is the raw, noisy time stream that the telescope would observe.(1024,). Target (Y)$T_{ng}$ (RFI Signal) is the isolated, non-Gaussian component you want the network to extract.(1024,). Data is formed inside signalsimulation.py script which generates each pair as follows: Generate $T_g$ (Sky Signal): This is the weak, Gaussian-distributed signal (astronomical noise, background). Generate $T_{ng}$ (RFI Signal): This is the strong, non-Gaussian, compressive burst (e.g., sinusoid in a Gaussian envelope). Calculate Input $S$: $S = T_g + T_{ng}$. For every single time sample in your dataset, the network is given $S$ and trained to minimize the difference with $T_{ng}$ Because we are training for RFI recovery, the U-Net model's output, $\hat{T}_{ng}$, is an estimate of the RFI. The final, clean astronomical signal ($\hat{T}_g$) is not the network's direct output but is calculated afterward as a Residual:$$\text{Clean Signal} (\hat{T}_g) = \text{Input Signal} (S) - \text{Recovered RFI} (\hat{T}_{ng})$$


To run the code (On Ilifu).
Fisrt Simulate the Astronomical Signal
```bash
singularity exec /idia/software/containers/ASTRO-GPU-PyTorch-2023-10-10.sif python signalsimulation.py
```
And then run the ML script for training the the model
```bash
singularity exec /idia/software/containers/ASTRO-GPU-PyTorch-2023-10-10.sif python ml.py
```
Once the training is finished, Results plot are saved to training_results/rfi_recovery_examples.png

If you wish to test if the RFI region is clean, you can run the gassianity tests:
```bash
singularity exec /idia/software/containers/ASTRO-PY3.simg python tests/resdual_test.py
```