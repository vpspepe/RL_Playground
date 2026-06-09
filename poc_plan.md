# PoC Architecture: Zero-Shot Latent Offline RL with Pre-trained Tadpole Encoder

## 1. Executive Summary & Intent
The objective of this Proof of Concept (PoC) is to validate if an Offline Reinforcement Learning agent can learn an optimal Tuned Mass Damper (TMD) placement policy by operating over a compressed physics feature space. Instead of training a geometric representation from scratch, this architecture utilizes a **pre-trained, frozen Tadpole Foundation Model Encoder** as a *Zero-Shot Feature Extractor*. 

By mapping the 10,000-point Fibonacci sphere differentials directly into the Tadpole latent space, we decouple field dimension reduction from policy optimization. The trainable cuckoo-policy component remains a lightweight Multi-Layer Perceptron (MLP).

---

## 2. Mathematical Formulation & Pre-processing
The input fields leverage the relative normalized differential framework already established in the repository's post-processing pipelines. This guarantees scale-invariance across highly disparate structural modes.

### 2.1 Differential Fields Optimization
For each simulation dataset $i$ at an excitation frequency $f$:
$$\text{vel\_diff}_i = \frac{\text{vel\_abs}_i - \text{vel\_base}_f}{\text{vel\_base\_max}_f}$$
$$\text{sti\_diff}_i = \frac{\text{sti\_abs}_i - \text{sti\_base}_f}{\text{sti\_base\_max}_f}$$

Where:
* $\text{vel\_base}_f, \text{sti\_base}_f$ are the un-damped harmonic baseline vectors from `base_sim_freq_*.npz`.
* $\text{vel\_base\_max}_f, \text{sti\_base\_max}_f$ are frequency-specific normalization constants.

---

## 3. The Spatial Voxelization Bridge
### 3.1 The Dimensionality Obstacle
The pre-trained **Tadpole Encoder backbone is a 3D Convolutional Neural Network (3D-CNN)** expecting structured voxel tensors of shape:
$$\mathbf{X}_{\text{tadpole}} \in \mathbb{R}^{B \times C \times D \times H \times W}$$
where $C=2$ (Channels: `vel_diff`, `sti_diff`), and $D, H, W$ represent spatial voxel dimensions.

However, the Marco dataset supplies unstructured arrays of shape `(10000,)` ordered linearly by the Fibonacci spiral sequence. Inserting raw flat lists directly into a 3D-CNN causes immediate shape breakdown.

### 3.2 Discrete Voxel Grid Mapping
To resolve this without altering the foundation model parameters, the flat sequence must pass through a spatial quantization layer:

1. **Bounding Box Quantization:** Define a static rectangular regular grid matching the spatial boundaries of the lower hemisphere using a specified spatial resolution (e.g., $32 \times 32 \times 16$).
2. **Coordinate Interpolation/Binning:** Retrieve the static global $(X, Y, Z)$ spatial positions of the 10,000 Fibonacci points. Normalize these dimensions to uniform array indices $[0..31, 0..31, 0..15]$.
3. **Tensor Assembly:** Map each index $k \in [0, 9999]$ to its closest grid voxel cell $(d, h, w)$. Assign the values of $\text{vel\_diff}[k]$ and $\text{sti\_diff}[k]$ to the respective channels of that target coordinate cell, filling unmapped background spaces with zeros.

---

## 4. MDP Specification (Offline Dataset Formatting)
The transitions extracted from historical files under `normalized_data/` are compiled sequentially to populate the static training replay buffer $\mathcal{D}$.

```
    +-------------------------------------------------+
              |  Raw Fibonacci Diff Vectors (vel_diff, sti_diff) |
              +-----------------------+-------------------------+
                                      |
                                      v (Voxelization Step)
              +-------------------------------------------------+
              |      Structured 3D Voxel Grid Tensor            |
              +-----------------------+-------------------------+
                                      |
                                      v (Pass through)
         =============================================================
         ======= FROZEN TADPOLE FOUNDATION MODEL ENCODER (eval) =======
         =============================+===============================
                                      |
                                      v
                           +--------------------+
                           |   Latent Vector z  |
                           +----------+---------+
                                      |
                                      v (Concatenation)


+-----------------------------------------+------------------------------------------+ | State Vector s: [ Latent z (64d) || Freq (1d) || TMD Coords (3d) || TMD Mass (1d) ] | +-----------------------------------------+------------------------------------------+ | v +---------------------------------+ | TRAINABLE AGENT POLICY (MLP) | +---------------------------------+

```

### 4.1 State Space ($S$):

The state vector groups the localized abstract kinematics with the macro parameters of the environment: $$s = [z \mathbin{\Vert} f \mathbin{\Vert} x_{\text{tmd}} \mathbin{\Vert} y_{\text{tmd}} \mathbin{\Vert} z_{\text{tmd}} \mathbin{\Vert} m_d]$$
Where: * $z \in \mathbb{R}^{64}$ is the zero-shot latent output vector from the frozen Tadpole encoder. * $f$ is the target excitation eigenfrequency. * $x_{\text{tmd}}, y_{\text{tmd}}, z_{\text{tmd}}, m_d$ indicate the current parameter location and magnitude metrics of the active configuration. ### 4.2 Action Space ($A$) Actions represent changes between structural evaluation pairs found inside Marco's logs: $$a = [\Delta x, \Delta y, \Delta z, \Delta m_d] \in [-1, 1]^4$$

### 4.3 Reward Function ($R$)
To match the objective of minimizing overall acoustic radiation without accessing a real-time solver, the delta change in global Equivalent Radiated Power (ERP) yields the immediate reward token: $$r = \frac{\text{ERP}_i - \text{ERP}_j}{\text{ERP}_{\text{base}}}$$ An action shifting the dampening parameters to a design space state that drops the local vibration output results in a linear positive return increment. 

---
## 5. R&D Rationale (Review Guidelines Compliance) * **Efficiency:** 
High computational execution savings. The multi-dimensional dataset is compressed *once* offline. The downstream Reinforcement Learning policy optimizes its weights over low-dimensional tensors ($69 \rightarrow 4$), eliminating heavy tensor backpropagation over the field network during training. * **Mathematical Soundness:** The implementation of the Behavior Cloning regularization constraint (via **TD3-BC** or **IQL**) locks the policy update trajectory inside the safe support boundary of the pre-calculated dataset buffer, stopping out-of-distribution value function explosion. * **SOTA Alignment:** Operates under current Scientific AI paradigms by transforming complex PDE solution fields into compressed representations via an existing scientific foundation backbone encoder model. * **Complexity Mitigation:** Prevents architectural debt. Since the pipeline decouples representation learning from control layout, the entire system can switch from the regularized Fibonacci sphere into a raw mesh Graph Neural Network (GNN) later without changing the policy model interface configuration.