Perfect! Here's the **complete design summary block** for your **U-Net + Flow Matching + Classifier-Free Guidance (CFM)** model trained on **CIFAR-10**, including architecture and **training hyperparameters**.

------

## 🧱 U-Net + FM + CFM on CIFAR-10 — Design Summary

### 🖼️ **Model Structure**

| Component           | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| **Input**           | Image `x ∈ ℝ^{3×32×32}`, time `t ∈ [0, 1]`, label `y ∈ {0, ..., 9}` or `None` |
| **Time Embedding**  | Fourier features → MLP → `t_emb ∈ ℝ^{t_dim}`                 |
| **Class Embedding** | `nn.Embedding(num_classes, y_dim)` → MLP → `y_emb ∈ ℝ^{y_dim}` |
| **Conditioning**    | Inject `t_emb` and `y_emb` into every residual block         |

------

### 🔧 **U-Net Architecture (Residual)**

```
Input: 3×32×32 image

↓ Conv2d(3 → 64)

↓ Encoder 1 (32×32 → 16×16)
   ResBlock(64→64, t_emb, y_emb)
   ResBlock(64→128, t_emb, y_emb)
   Downsample (stride=2)

↓ Encoder 2 (16×16 → 8×8)
   ResBlock(128→128)
   ResBlock(128→256, t_emb, y_emb)
   Downsample

↓ Encoder 3 (8×8 → 4×4)
   ResBlock(256→256)
   ResBlock(256→512, t_emb, y_emb)
   Downsample

↓ Bottleneck (4×4)
   ResBlock(512→512, t_emb, y_emb)
   ResBlock(512→512, t_emb, y_emb)

↑ Decoder 1 (4×4 → 8×8)
   Upsample
   Concat skip
   ResBlock(1024→512)
   ResBlock(512→256, t_emb, y_emb)

↑ Decoder 2 (8×8 → 16×16)
   Upsample
   Concat skip
   ResBlock(512→256)
   ResBlock(256→128, t_emb, y_emb)

↑ Decoder 3 (16×16 → 32×32)
   Upsample
   Concat skip
   ResBlock(256→128)
   ResBlock(128→64, t_emb, y_emb)

↑ Final Conv2d(64 → 3)
Output: velocity field v(x, t, y) ∈ ℝ^{3×32×32}
```

>   Residual block = GroupNorm + SiLU + Conv + time/label conditioning + residual connection

------

### ⚙️ **Training Hyperparameters**

| Hyperparameter    | Recommended Value                | Notes                                      |
| ----------------- | -------------------------------- | ------------------------------------------ |
| **Batch size**    | `64` (or `128` if memory allows) | Tradeoff between stability and speed       |
| **Learning rate** | `1e-4` (use Adam)                | Try `5e-5` if training unstable            |
| **Optimizer**     | `Adam(β1=0.9, β2=0.999)`         | Standard setup for generative training     |
| **Weight decay**  | `0.0` or `1e-5`                  | Optional; small value helps regularization |
| **Gradient clip** | `1.0`                            | Helps prevent spikes in training           |
| **LR scheduler**  | Fixed or Cosine Annealing        | Cosine can help late-stage stability       |
| **EMA**           | `0.999` decay                    | Optional but helps final sampling quality  |

------

### ⏱️ **Training Schedule**

| Setting             | Value                       | Notes                            |
| ------------------- | --------------------------- | -------------------------------- |
| **Max Epochs**      | `500–1000`                  | For full convergence on CIFAR-10 |
| **Steps per Epoch** | `50,000 / batch_size ≈ 782` | Batch size = 64                  |
| **Early Stopping**  | `patience = 20 epochs`      | Monitor val loss or FID          |
| **Min Δ (delta)**   | `1e-4`                      | For stopping criterion           |

------

### 🧪 **Classifier-Free Guidance (CFM)**

| Component                | Setting                                                      | Notes                              |
| ------------------------ | ------------------------------------------------------------ | ---------------------------------- |
| **Dropout Rate**         | `10%–20%` of label embeddings                                | Drop `y` at random during training |
| **Guidance Scale**       | `2.0 – 6.0`                                                  | Tune at inference time             |
| **Conditioned Sampling** | v(x, t, y) = v(x, t, null) + scale × (v(x, t, y) - v(x, t, null)) | Standard CFG formula               |

------

### 📌 Summary Recommendations

-   ✅ Use **residual blocks** throughout with time and label embedding
-   ✅ Normalize input images to `[-1, 1]`
-   ✅ Use **U-Net with 3 downsampling stages** (sufficient for 32×32)
-   ✅ Train for 500k–1M iterations with **EMA**, **Adam**, and **CFM dropout**
-   ✅ Use fixed LR or cosine decay, and monitor val loss or FID for early stopping

------

Let me know if you'd like a full PyTorch training loop or sampling script based on this setup!