# ECEC247A Final Project - Predicting Keystrokes from Electromyography Signals

This project explores deep learning models for predicting QWERTY keystrokes from surface electromyography (sEMG) signals using the **emg2qwerty dataset**. The goal is to decode text typed by a user based only on muscle activity recorded from wrist sensors.

We evaluate multiple neural network architectures including CNNs, RNNs, LSTMs, GRUs, and hybrid CNN + recurrent models. Our experiments show that combining convolutional feature extraction with recurrent sequence modeling gives the best performance.

The best model in this project is a **CNN + LSTM hybrid**, achieving a **Character Error Rate (CER) of 18.0** on the test set.

## Authors
- Andrew Wild
- Aaron Lit
- Joanna Tang
- Jimit Bhalavat

## Dataset

We use the **emg2qwerty dataset**, which contains sEMG recordings from wrist electrodes while users type on a QWERTY keyboard.

Key details:

- 32 electrode channels (16 per wrist)
- Sampling rate: **2 kHz**
- Data collected from typing sessions
- Train / validation / test splits provided in the original repository

In this project we use **single-subject training** with data from subject `89335547`.

## Input Representation

Raw EMG signals are converted into **log spectrograms** using a Short-Time Fourier Transform.

Parameters used:

- `nfft = 64`
- `hop length = 16`
- Frame every **8 ms**
- **33 frequency bins per channel**

During training, recordings are split into windows:

- Window length: **2000 samples (4 seconds)**
- Context: **900 ms before** and **100 ms after**

The extra context helps the model use surrounding temporal information.

## Training Augmentations

Several augmentations are used to improve model robustness.

- Random Band Rotation: Rotates electrode channels within each wrist band, simulates small changes in band placement
- Temporal Alignment Jitter: Shifts timing between left and right wrist signals, simulates synchronization differences between sensors
- SpecAugment: Randomly masks small regions in time and frequency
- Gaussian Noise: Adds small noise to the signal
- Input Normalization: Normalizes signal amplitude before model input

In the single-subject setting, some augmentations (such as band rotation) can hurt performance because the same user wears the sensors consistently.

## Model Architectures

All models share the same preprocessing pipeline:

- Log-spectrogram input
- Spectrogram normalization
- Feature MLP
- CTC output layer

Each architecture receives a **768-dimensional feature vector per time step** and is trained using **CTC loss with greedy decoding**.

#### CNN Baseline

The baseline model is a **Time-Depth Separable (TDS) convolutional network**. The CNN learns temporal patterns directly from spectrogram features using convolutional layers. It captures local temporal structure while remaining computationally efficient. We also test additional variants of the CNN baseline using Gaussian noise injection and Input normalization These experiments test whether simple signal-level regularization improves performance without changing the architecture.

#### Vanilla RNN

We evaluate a simple recurrent neural network as a baseline sequence model. Vanilla RNNs often suffer from **vanishing or exploding gradients** when sequences are long. To stabilize training we apply gradient clipping using **Pascanu’s norm-based criterion**, weight decay, and dropout

We experiment with `tanh` and `ReLU` nonlinearities, multiple layers, and different hidden sizes. This model serves as a baseline for comparing gated recurrent architectures.

#### LSTM

We evaluate a **Long Short-Term Memory (LSTM)** network. LSTMs introduce gating mechanisms that help preserve gradient flow and capture long-range temporal dependencies. This makes them well suited for EMG sequences.Experiments vary hidden size (192–256) and number of layers (2–3)

#### GRU

We also test **Gated Recurrent Units (GRUs)**. GRUs are similar to LSTMs but use fewer parameters. They merge some gating operations and remove the separate memory cell. Gradient clipping and dropout are used to stabilize training.

#### CNN + RNN Hybrid

This model combines convolutional feature extraction with a vanilla RNN.The CNN layers first extract local temporal patterns from the spectrogram. The RNN then models longer temporal dependencies. Additional augmentations tested gaussian noise and specAugment

#### CNN + LSTM Hybrid

This architecture uses the same CNN front-end but replaces the RNN with an LSTM. The CNN captures local patterns while the LSTM models long-range temporal dependencies. This model achieved the **best results** in our experiments.

#### CNN + GRU Hybrid

The final hybrid architecture replaces the LSTM with a GRU. This tests whether a simpler gated model can achieve similar performance when CNN layers already capture local structure.

## Training Setup

Models are trained using:

- **Optimizer:** Adam
- **Learning rate:** `2e-4 – 1e-3`
- **Batch size:** usually 16
- **Training epochs:** 25–60
- **Loss:** CTC

Performance is evaluated using **Character Error Rate (CER)**. Lower CER means the predicted text matches the true keystrokes more closely.

## Results

Best results achieved for each architecture:

| Architecture | Best Test CER |
|---|---|
| CNN baseline | 24.9 |
| CNN + noise + normalization | 20.8 |
| Vanilla RNN | 28.9 |
| LSTM | 20.8 |
| GRU | 81.4 |
| CNN + RNN | 23.4 |
| CNN + GRU | 100.0 |
| CNN + LSTM | **18.0** |

The **CNN + LSTM hybrid** performs best because CNN extracts local spectrogram features and LSTM captures long-range temporal dependencies
