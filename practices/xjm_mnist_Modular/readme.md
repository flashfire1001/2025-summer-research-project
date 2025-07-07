## 📁 Project Structure Overview

This project implements a conditional generative model trained on the MNIST dataset using classifier-free guidance (CFG) and a Gaussian diffusion process. Below is a breakdown of each module in the codebase:

---

### `main.py`

The main script that orchestrates the training and generation pipeline. It initializes the MNIST sampler, constructs the conditional probability path, builds the UNet model, trains it using the CFGTrainer, and finally simulates and visualizes generated samples under varying guidance scales. This file serves as the primary entry point for experimentation.

---

### `config_common.py`

Contains shared configuration and utility functions. Defines the computing device (`cuda` or `cpu`) and a helper function `model_size_b()` for estimating the memory footprint of a model. This centralized configuration ensures consistency and simplifies device-agnostic programming throughout the codebase.

---

### `sampler.py`

Defines the data sampling abstraction used throughout the project. The `Sampleable` base class enforces a standard interface for distributions. `IsotropicGaussian` samples from a simple Gaussian prior, while `MNISTSampler` handles loading, preprocessing, and random sampling from the MNIST dataset. These classes are essential for training and sampling workflows.

---

### `model.py`

Implements the neural network architecture used to approximate the conditional vector field. The core model is a UNet-like structure enhanced with Fourier time encodings and label embeddings. Residual layers are used throughout to stabilize training and enable the model to condition effectively on both time and class labels.

---

### `trainer.py`

Provides training routines for the vector field model. Includes a base `Trainer` class and a specialized `CFGTrainer` for classifier-free guidance. The trainer handles optimization, label masking (null-label replacement), and loss computation. It encapsulates all logic needed to fit the model to the conditional path data.

---

### `ode_path.py`

Contains core diffusion logic, including time-dependent interpolation functions (`Alpha`, `Beta`), the `GaussianConditionalProbabilityPath`, and the `CFGVectorFieldODE`. These classes define the mathematical structure of the generative process, including forward sampling, velocity field construction, and analytical score functions for training and simulation.

---

### `simulator.py`

Defines the numerical simulation process used to generate samples from the trained model. Includes a base `Simulator` interface and an `EulerSimulator` implementation. This module integrates the ODE over time using the learned vector field to transform Gaussian noise into data-like samples. It also supports trajectory visualization for debugging or inspection.

251cx