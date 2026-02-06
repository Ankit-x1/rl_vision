# Latency-Aware Vision Model for Edge Devices using Reinforcement Learning

**A dynamic inference system that learns to balance accuracy, latency, and compute constraints using RL-controlled early exits.**

---

##  Why This Problem Matters

Edge devices (mobile phones, IoT sensors, embedded systems) cannot afford to run full deep neural network inference for every input:

- **Limited compute budget** (battery, thermal constraints)
- **Real-time requirements** (autonomous systems, robotics)
- **Heterogeneous workloads** (easy vs hard samples)

**The Core Insight:**
Not all images require the same computational effort. A clear image of a cat doesn't need 50 layers — but a blurry, occluded object might.

Traditional CNNs use **fixed depth** for all inputs. This project uses **Reinforcement Learning** to learn a policy that decides:


##  What Makes This Novel

### 1. **Dynamic Inference via Early Exits**
- ResNet-18 backbone with **intermediate classifiers** at multiple depths
- Each exit head can produce a prediction
- RL agent decides when to stop

### 2. **Multi-Objective Optimization**
The RL agent optimizes:
```
Reward = α × Accuracy - β × Latency - γ × Compute
```
This is **RLHF-adjacent**: learning from multi-dimensional feedback, not just cross-entropy loss.

### 3. **Edge-Aware Training**
Simulates realistic edge constraints:
- Target latency budgets (e.g., 20ms)
- Layer-wise compute costs
- Device profiles (low/medium/high power)

### 4. **Adaptive Behavior**
The same model can adapt to different deployment scenarios:
- **Aggressive mode**: Exit early, prioritize speed
- **Conservative mode**: Go deeper, prioritize accuracy
- **Balanced mode**: Learn optimal tradeoff

---

##  Architecture

### Base Model: ResNet-18 with Early Exits

```
Input Image (32×32×3)
    ↓
[Conv Block 1] → Exit Head 1 (optional prediction)
    ↓
[Conv Block 2] → Exit Head 2 (optional prediction)
    ↓
[Conv Block 3] → Exit Head 3 (optional prediction)
    ↓
[Conv Block 4] → Final Classifier (always available)
```

### RL Environment

**State Space:**
- Confidence (entropy) of current exit head
- Current layer index
- Cumulative latency so far
- Compute budget remaining

**Action Space:**
- `CONTINUE`: Proceed to next layer
- `EXIT`: Use current prediction

**Reward Function:**
```python
reward = correct_prediction - λ_latency × latency - λ_compute × compute_used
```

### RL Algorithm: Proximal Policy Optimization (PPO)

- **Policy Network**: Maps state → action probabilities
- **Value Network**: Estimates expected return
- **Training**: On-policy, stable, sample-efficient

---

## 📊 Dataset

**CIFAR-100** (100 classes, 60,000 images)
- More challenging than CIFAR-10
- Diverse enough to show adaptive behavior
- Small enough for rapid iteration

**Why not ImageNet?**
- CIFAR-100 is sufficient to demonstrate the concept
- Faster training cycles
- Easier to analyze failure modes

---

##  Getting Started

### Installation

```bash
# Clone the repository
cd vision_rl_edge

# Install dependencies
pip install -r requirements.txt
```

### Training Pipeline

#### Step 1: Train Supervised Base Model
```bash
python train_supervised.py --config configs/model_config.yaml
```
This trains the ResNet-18 backbone with all early-exit heads using standard cross-entropy loss.

#### Step 2: Train RL Policy
```bash
python train_rl.py --config configs/rl_config.yaml --device_profile low_power
```
This trains the PPO agent to learn when to exit.

#### Step 3: Evaluate
```bash
python evaluate.py --model_path checkpoints/rl_model.pth --device_profile low_power
```

---

## 📈 Expected Results

### Metrics We Track

1. **Accuracy vs Latency Curve**
   - Compare RL policy vs always-exit-early vs always-exit-late
   
2. **Compute Efficiency**
   - FLOPs saved vs accuracy drop
   
3. **Adaptive Behavior**
   - Exit distribution across difficulty levels
   
4. **Reward Curves**
   - PPO training stability and convergence

### Visualization Examples

```
results/
├── latency_vs_accuracy.png       # Pareto frontier
├── exit_distribution.png         # Which layers are used most
├── reward_curve.png               # RL training progress
├── confidence_analysis.png        # Entropy vs exit decision
```

---

##  What I Learned (Key Insights)

### 1. **Tradeoffs Are Real**
- Early exits save compute but hurt accuracy on hard samples
- RL learns to identify "easy" vs "hard" inputs

### 2. **Reward Engineering Matters**
- Balancing α, β, γ is critical
- Too much latency penalty → always exit early (bad accuracy)
- Too little → never exit early (no speedup)

### 3. **When RL Helps vs Hurts**
- **Helps**: When input difficulty varies (heterogeneous data)
- **Hurts**: When all inputs are equally hard (RL overhead not worth it)

### 4. **Failure Modes**
- **Overconfident exits**: Model exits early on hard samples
- **Underconfident exits**: Model never exits, defeats purpose
- **Solution**: Calibration techniques (temperature scaling)

---

## 🔧 Configuration

### Device Profiles (`configs/edge_config.yaml`)

```yaml
low_power:
  max_latency_ms: 15
  max_flops: 50M
  target_accuracy: 0.65

medium_power:
  max_latency_ms: 30
  max_flops: 150M
  target_accuracy: 0.75

high_power:
  max_latency_ms: 50
  max_flops: 500M
  target_accuracy: 0.85
```

### RL Hyperparameters (`configs/rl_config.yaml`)

```yaml
ppo:
  learning_rate: 3e-4
  gamma: 0.99
  clip_epsilon: 0.2
  epochs_per_update: 10
  batch_size: 64

reward_weights:
  alpha: 1.0      # Accuracy weight
  beta: 0.5       # Latency penalty
  gamma: 0.3      # Compute penalty
```

---

##  Connections to Research

This project builds on:

1. **BranchyNet** (2016): First work on early exits
2. **MSDNet** (2018): Multi-scale dense networks with anytime prediction
3. **Dynamic Neural Networks** (Survey 2021): Adaptive inference
4. **RLHF** (2022+): Multi-objective optimization with RL

**Novel Contribution:**
Explicit edge-aware training with simulated latency/compute constraints.

---

## Later/ Future Extensions

### 1. **Real Hardware Deployment**
- Deploy on Raspberry Pi / Jetson Nano
- Measure actual latency (not simulated)

### 2. **Vision Transformers**
- Replace ResNet with ViT
- Dynamic token pruning

### 3. **Federated Learning**
- Train RL policy across multiple edge devices
- Personalized exit strategies

### 4. **Energy Optimization**
- Add battery drain to reward function
- Learn energy-aware policies

---

##  Contributing

This is a research project. If you find bugs or have ideas for improvements:
- Open an issue
- Submit a pull request
- Reach out for collaboration

---



##  Acknowledgments

- **CIFAR-100 Dataset**: Learning Multiple Layers of Features from Tiny Images (Krizhevsky, 2009)
- **PPO Algorithm**: Proximal Policy Optimization Algorithms (Schulman et al., 2017)
- **Early Exit Networks**: BranchyNet (Teerapittayanon et al., 2016)


**Status:**  Fully implemented and documented
