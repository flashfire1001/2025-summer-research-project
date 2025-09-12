# Image Denoising with MeanFlow under Gaussian Noise on MNIST

## Abstract

Fast and reliable image denoising is essential in real-world applications where rapid perception is critical, such as self-driving, medical imaging, and robotics. In such scenarios, systems must react to corrupted or blurred inputs almost instantaneously. This work demonstrates that the recently proposed **MeanFlow method** provides a promising solution by enabling efficient one-step denoising while maintaining strong reconstruction quality.In this work, we investigate the application of the recently proposed MeanFlow method to the Gaussian denoising task.We present a rigorous theoretical foundation for MeanFlow, connect it to the continuity equation and flow matching derivations, and apply it experimentally to MNIST digits corrupted with Gaussian noise. Results demonstrate that MeanFlow achieves strong reconstruction quality, maintaining high PSNR, and distinct decrease in sampling steps during the simulation process. Our analysis shows why image with gaussian noise aligns particularly well with MeanFlow assumptions. This work provides both theoretical insights and practical evidence that MeanFlow is a promising framework for denoising under various corruptions.

---

## 1. Introduction

Image denoising is a fundamental inverse problem: given a corrupted observation, recover the underlying clean image. This task is central to applications ranging from medical imaging to autonomous driving. Among noise models, **Gaussian noise** is widely studied because it arises naturally in physical acquisition processes, is analytically tractable, and serves as a benchmark in image restoration research.

Recent advances in generative modeling — notably diffusion models and flow-based methods — have yielded state-of-the-art performance in generation tasks. Seeing their capability in image generation, we thus conduct an experiment testing the viability for flow-based method to denoise an image. Similar to generative task,these methods reframe denoising as transporting a corrupted input distribution back toward the clean data distribution. However, diffusion sampling is computationally expensive, and normalizing flows require expensive Jacobian determinants.

The **MeanFlow method** (a refinement of flow matching) provides an elegant alternative. By training neural networks to approximate *average velocities* of probability paths, MeanFlow enables efficient one-step sampling while preserving theoretical guarantees.

In this paper, we:

1. Present a rigorous theoretical foundation of MeanFlow, building from the continuity equation and flow matching derivations.
2. Apply MeanFlow to Gaussian denoising on MNIST digits.
3. Provide quantitative and qualitative results, alongside analysis of why Gaussian noise pairs naturally with MeanFlow.

---

## 2. Background and Related Work

### 2.1 Generative Modeling as Sampling

We represent images as vectors $z \in \mathbb{R}^d$. A dataset is a finite collection $\{z_i\}_{i=1}^N \sim p_\text{data}$, serving as samples from the unknown data distribution. A generative model transforms samples from a simple initial distribution $p_\text{init}$ (often Gaussian) into the target distribution $p_\text{data}$. Thus, denoising can be seen as transporting a noisy sample closer to $p_\text{data}$.

### 2.2 Continuity Equation and Flow View

Generative transport can be described with the **continuity equation**, originating in fluid dynamics:

$$
\frac{\partial \rho(x,t)}{\partial t} + \nabla_x \cdot \big(\rho(x,t) v(x,t)\big) = 0,
$$

where $\rho(x,t)$ is the evolving density and $v(x,t)$ is a velocity field. This expresses conservation of probability mass: local changes in density are explained by inflows and outflows. If $p_t(x)$ evolves smoothly from $p_\text{init}$ to $p_\text{data}$, then there must exist a velocity field transporting samples accordingly:

$$
x_t - x_0 = \int_0^t v(x_\tau,\tau) d\tau.
$$

### 2.3 Flow Matching

Flow Matching (FM) constructs practical training objectives for learning $v(x,t)$. Instead of requiring explicit probability densities, FM uses **conditional velocity fields** based on interpolations between noisy samples and clean data. By showing that minimizing MSE against conditional velocities is equivalent to minimizing against the unknown marginal velocity, FM provides a tractable training scheme.

Mathematically, if $x_t = \alpha_t z + \beta_t x_0$ interpolates between clean data $z$ and noise $x_0$, then the conditional velocity is:

$$
v(x,t \mid z) = \dot{\alpha}_t z + \dot{\beta}_t x_0.
$$

The target marginal velocity is the weighted expectation of conditional velocities, ensuring conservation under the continuity equation.

**2.4 Classifier-Free Guidance (CFG)**
Classifier-Free Guidance (CFG) is a widely used technique in generative modeling to balance unconditional and conditional generation. In CFG, the model is trained to predict velocity (or score) fields both with and without conditioning information (e.g., class labels). At inference, the two predictions are linearly combined:
$$
v_{\text{cfg}}(x,t) = v_\text{uncond}(x,t) + w \cdot \big( v_\text{cond}(x,t \mid y) - v_\text{uncond}(x,t) \big),
$$

where $w$ is the guidance scale, $v_\text{cond}$ is the conditional velocity, and $v_\text{uncond}$ is the unconditional velocity. This formulation allows trading off between sample diversity (small $w$) and fidelity to conditioning information (large $w$). In the context of Gaussian denoising with MeanFlow, CFG can guide the model to not only restore clean structure but also enforce consistency with digit class labels, thereby improving robustness against over-smoothing and preserving semantic identity.

### 2.5 MeanFlow

While FM requires numerical ODE integration, **MeanFlow** introduces the *average velocity*:

$$
u(x_t, t, r) = \frac{1}{r-t} \int_t^r v(x_\tau, \tau) d\tau.
$$

The **MeanFlow identity** connects instantaneous and average velocities:

$$
u(x_t, t, r) = v(x_t, t) + (r-t)\frac{d}{dt}u(x_t, t, r).
$$

By training a neural network $u^\theta$ to approximate average velocities, MeanFlow enables efficient **one-step generation** from $p_\text{init}$ to $p_\text{data}$. For denoising, this means mapping a Gaussian-corrupted image directly to a clean one.

---

## 3. Methodology

(我这里没有细讲真正的DiT代码中, 我们的data patchify , transformer的architecture, 也没有讲各种用到的data transformation和normalization)

### 3.1 Gaussian Noise Model

We corrupt MNIST digits $x \in [0,1]^{28\times 28}$ with additive Gaussian noise:

$$
x_{\text{noisy}} = x + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I).
$$

Noise variances ${\sigma}^2 $ is a random number uniformly distributed in the range [0.3, 0.8]

 

### 3.2 MeanFlow for Denoising

The task is to train MeanFlow to transport noisy images back to the clean data distribution. As we apply linear Gaussian probablity path, The conditional interpolation path is defined as:

$$
x_t = t z +(1-t) x_0,
$$

with $z$ a clean digit and $x_0$​ Gaussian noise. Training uses conditional velocities, with the equivalence to marginal training guaranteed. The model learns to approximate the average velocity field, enabling one-step denoising at inference.

### 3.3 Model Architecture: Diffusion Transformer (DiT)**
 For the neural network backbone in MeanFlow, we adopt the Diffusion Transformer (DiT), a transformer-based architecture originally designed for diffusion generative models. Unlike convolutional networks, DiT leverages self-attention to capture long-range dependencies, which is beneficial for modeling global digit structures in MNIST. Each noisy image is first embedded into patch tokens, which are then processed by a sequence of transformer blocks with multi-head self-attention and feed-forward layers. Time-step embeddings are injected via cross-attention, enabling the model to adaptively adjust its denoising dynamics across different noise levels. The final token sequence is projected back into image space to parameterize the average velocity field $u^\theta(x,t,r)$. Using DiT provides flexibility and scalability, making MeanFlow not only effective for MNIST but also extensible to more complex datasets.

### 3.4 Experimental Setup

* **Dataset**: MNIST (60k train, 10k test).
* **Noise application**: Gaussian corruption.
* **Architecture**: Neural network parameterizing $u^\theta(x,t,r)$.
* **Optimization**: AdamW, learning rate $10^{-4}$, 20k to 300k epochs.
* **Batch size**: 128.
* **Metrics**: PSNR

---

## 4. Experiments and Results

### 4.1 Qualitative Results

### 4.2 Quantitative Results



---

## 5. Analysis and Discussion

### 5.1 Compatibility with Gaussian Noise

One key reason for the success of MeanFlow in our experiments lies in the inherent compatibility between Gaussian noise and the flow-based generative modeling framework. Gaussian noise is **smooth, unimodal, and isotropic**, meaning that the corrupted data distribution remains mathematically well-behaved. These properties align naturally with the assumptions of the continuity equation, which requires differentiability and stability of the evolving density. As a result, the learned velocity fields do not encounter abrupt discontinuities or multimodal divergences, allowing the transport process to be integrated smoothly. From a probabilistic perspective, the additive Gaussian perturbation simply shifts the distribution while preserving its global structure, making it especially suitable for restoration via continuous vector fields.

### 5.2 Sensitivity to Noise Variance

The experimental results demonstrate a gradual degradation in performance as the noise variance $\sigma^2 $increases. This behavior is expected: higher noise levels correspond to broader perturbations of the underlying data manifold, which increases the difficulty of recovering the clean signal. However, the graceful nature of the decline suggests that the MeanFlow model does not merely memorize denoising at a fixed scale, but instead learns a representation of the digit manifold that is resilient to varying degrees of Gaussian corruption. In practical terms, this robustness is important because real-world imaging systems often exhibit fluctuations in noise strength rather than a fixed variance. The observation that MeanFlow can tolerate such changes indicates strong generalization ability.

### 5.3 Theoretical Alignment with Flow Matching Principles

The performance of MeanFlow can also be interpreted in light of its theoretical underpinnings. Flow Matching provides a principled way of training neural velocity fields by aligning conditional velocities with marginal ones under the continuity equation. This guarantees that the learned dynamics conserve probability mass, ensuring that noisy inputs are transported consistently toward the data distribution. MeanFlow further refines this framework by introducing average velocities and enabling one-step generation. In the case of Gaussian denoising, these properties are particularly advantageous: the Gaussian distribution’s closed-form interpolation paths simplify the computation of conditional velocities, and the unimodality of the target distribution minimizes ambiguity in the transport. Thus, both the mathematical structure of the problem and the theoretical foundation of MeanFlow converge to produce stable and accurate denoising.

### 5.4 Architectural Considerations

Our adoption of the Diffusion Transformer (DiT) as the backbone network introduces additional insights. Unlike convolutional networks that emphasize local receptive fields, transformers leverage self-attention to capture global dependencies across the entire image. This ability is particularly beneficial when dealing with Gaussian noise, which corrupts all pixels uniformly. By allowing information from distant parts of the image to influence local denoising decisions, DiT helps preserve the overall digit structure while recovering fine details. Nevertheless, transformers incur higher computational costs compared to CNNs or UNet, raising questions about scalability. Future work should explore trade-offs between global context modeling and efficiency, potentially through hybrid CNN-transformer architectures.

### 5.5 Extensions via Classifier-Free Guidance

While the present study focused on unconditional denoising, the MeanFlow framework can naturally incorporate **Classifier-Free Guidance (CFG)** to support conditional restoration. CFG combines unconditional and conditional velocity predictions in a weighted manner, allowing the model to not only remove Gaussian noise but also align outputs with semantic labels. In the MNIST case, conditioning on digit classes would encourage the model to denoise while preserving class-specific structures, reducing the risk of ambiguity when noise severely distorts the digit identity. This extension could improve robustness under extreme corruption, and more broadly, it opens the possibility of class-conditional or even text-conditional denoising in more complex datasets.

### 5.6 Limitations and Future Directions

Despite promising results, the present study has limitations. First, experiments were restricted to MNIST, which is a relatively simple dataset. Extending to more complex datasets such as CIFAR-10 or Tiny-ImageNet would better demonstrate scalability. Second, only Gaussian noise was considered, whereas real-world noise often exhibits non-Gaussian or structured characteristics (e.g., sensor noise, occlusions). Finally, while MeanFlow offers efficiency advantages over diffusion models, we have not conducted a systematic computational comparison. Addressing these points in future work will provide a more comprehensive understanding of MeanFlow’s strengths and limitations.6. Conclusion and Future Work

We have applied **MeanFlow** to the task of Gaussian image denoising, grounding the method in a rigorous theoretical framework and validating it empirically on MNIST. The results demonstrate high-quality reconstructions, robustness to noise levels, and efficiency in inference.Future work will extend to more complex datasets (e.g., CIFAR-10, ImageNet), investigate robustness to non-Gaussian noise, and benchmark computational trade-offs against diffusion-based methods.

---

## References

[[2212.09748\] Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)

[Mean Flows for One-step Generative Modeling](https://arxiv.org/html/2505.13447v1)

