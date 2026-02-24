# Federated Learning Documentation for DeepSC

## Table of Contents
1. [Introduction to Federated Learning](#introduction-to-federated-learning)
2. [Overview of DeepSC](#overview-of-deepsc)
3. [Federated Learning Workflow](#federated-learning-workflow)
4. [Code Structure and Components](#code-structure-and-components)
5. [Detailed Code Walkthrough](#detailed-code-walkthrough)
6. [Key Concepts Explained](#key-concepts-explained)
7. [Running the Code](#running-the-code)

---

## Introduction to Federated Learning

### What is Federated Learning?

**Federated Learning (FL)** is a distributed machine learning approach where multiple clients (devices or parties) collaboratively train a model without sharing their raw data. Instead of sending data to a central server, the training happens locally on each client, and only model updates (weights/parameters) are sent to a central server for aggregation.

### Why Use Federated Learning?

1. **Privacy**: Raw data never leaves the client's device
2. **Efficiency**: Reduces bandwidth by sending only model updates instead of large datasets
3. **Scalability**: Can leverage many distributed devices
4. **Regulatory Compliance**: Helps meet data privacy regulations (GDPR, HIPAA, etc.)

### Traditional vs. Federated Learning

**Traditional Centralized Learning:**
```
┌─────────┐     ┌─────────┐     ┌─────────┐
│ Client1 │     │ Client2 │     │ Client3 │
│  Data   │────▶│  Data   │────▶│  Data   │
└─────────┘     └─────────┘     └─────────┘
     │               │               │
     └───────────────┼───────────────┘
                     ▼
              ┌─────────────┐
              │   Server    │
              │ (All Data)  │
              │   Train     │
              └─────────────┘
```

**Federated Learning:**
```
┌─────────┐     ┌─────────┐     ┌─────────┐
│ Client1 │     │ Client2 │     │ Client3 │
│  Data   │     │  Data   │     │  Data   │
│ Train   │     │ Train   │     │ Train   │
└────┬────┘     └────┬────┘     └────┬────┘
     │               │               │
     │ Model Updates │ Model Updates │ Model Updates
     └───────────────┼───────────────┘
                     ▼
              ┌─────────────┐
              │   Server    │
              │ Aggregate   │
              │   Updates   │
              └──────┬──────┘
                     │
              Global Model
```

### Key Federated Learning Concepts

1. **Server**: Central coordinator that aggregates model updates
2. **Clients**: Distributed devices/parties that hold local data
3. **Global Model**: The shared model maintained by the server
4. **Local Training**: Each client trains on its own data
5. **Aggregation**: Server combines updates from multiple clients (typically using FedAvg)
6. **Round**: One iteration of: select clients → local training → aggregation

---

## Overview of DeepSC

### What is DeepSC?

**DeepSC (Deep learning enabled Semantic Communication Systems)** is a neural network architecture designed for semantic communication over noisy wireless channels. Unlike traditional communication systems that transmit bits, DeepSC transmits semantic meaning, making communication more efficient and robust.

### DeepSC Architecture Flow

```
Input Sentence (text)
    │
    ▼
┌─────────────────┐
│ Semantic Encoder│  ← Transformer encoder (extracts meaning)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Channel Encoder │  ← Prepares for transmission
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Noisy Channel  │  ← AWGN, Rayleigh, or Rician fading
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Channel Decoder │  ← Recovers from noise
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Semantic Decoder│  ← Transformer decoder (reconstructs sentence)
└────────┬────────┘
         │
         ▼
Output Sentence (reconstructed text)
```

### Why Federated Learning for DeepSC?

- **Distributed Data**: Different clients may have different text data (different languages, domains, etc.)
- **Privacy**: Text data may contain sensitive information
- **Robustness**: Training on diverse distributed data improves generalization
- **Efficiency**: Clients can train locally without uploading large text datasets

---

## Federated Learning Workflow

### High-Level Workflow

The federated learning process follows these steps:

```
1. INITIALIZATION
   ├── Load vocabulary and datasets
   ├── Partition data across clients
   ├── Initialize global DeepSC model
   └── Create data loaders for each client

2. FEDERATED TRAINING LOOP (for each round)
   │
   ├── STEP 1: Client Selection
   │   └── Server randomly selects subset of clients
   │
   ├── STEP 2: Local Training (for each selected client)
   │   ├── Copy global model to client
   │   ├── Train on local data for E epochs
   │   ├── Sample channel noise for each epoch
   │   └── Return updated model weights
   │
   ├── STEP 3: Aggregation
   │   ├── Server receives model updates from clients
   │   ├── Compute weighted average (FedAvg)
   │   └── Update global model
   │
   └── STEP 4: Checkpointing (optional)
       └── Save model periodically

3. FINALIZATION
   └── Save final global model
```

### Detailed Round-by-Round Process

**Round 1:**
```
Server: Initialize global model with random weights
        └── Global Model: W₀ (random initialization)

Server: Select clients [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

Client 0: Receive W₀ → Train on local data → Return W₀¹
Client 1: Receive W₀ → Train on local data → Return W₁¹
Client 2: Receive W₀ → Train on local data → Return W₂¹
...
Client 9: Receive W₀ → Train on local data → Return W₉¹

Server: Aggregate: W₁ = FedAvg(W₀¹, W₁¹, ..., W₉¹)
```

**Round 2:**
```
Server: Select clients [2, 5, 7, 10, 12, 15, 18, 19, 3, 8]

Client 2: Receive W₁ → Train on local data → Return W₂²
Client 5: Receive W₁ → Train on local data → Return W₅²
...
Client 8: Receive W₁ → Train on local data → Return W₈²

Server: Aggregate: W₂ = FedAvg(W₂², W₅², ..., W₈²)
```

This continues for `--rounds` iterations.

---

## Code Structure and Components

### File Organization

```
DeepSC-master/
├── fl_train.py          # Main federated training script
├── fl_data.py           # Dataset loading utilities
├── fl_partition.py      # Data partitioning strategies
├── fl_eval.py           # Evaluation functions (BLEU score)
├── utils.py             # Training utilities, channel simulation
├── models/
│   ├── transceiver.py   # DeepSC model architecture
│   └── mutual_info.py   # Mutual information estimation
└── data/
    └── europarl/
        ├── train_data.pkl
        ├── test_data.pkl
        └── vocab.json
```

### Component Responsibilities

| File | Purpose |
|------|---------|
| `fl_train.py` | Orchestrates federated learning: client selection, local training, aggregation |
| `fl_data.py` | Loads preprocessed Europarl datasets (train/test splits) |
| `fl_partition.py` | Splits training data across clients (IID or non-IID) |
| `fl_eval.py` | Evaluates model using BLEU score on test set |
| `utils.py` | Training step, channel simulation (AWGN/Rayleigh/Rician), loss functions |
| `models/transceiver.py` | DeepSC neural network architecture (encoder-decoder) |

---

## Detailed Code Walkthrough

### 1. Initialization Phase (`fl_train.py` lines 167-244)

#### 1.1 Load Vocabulary (lines 167-190)

```python
vocab = json.load(open(vocab_path, "r", encoding="utf-8"))
token_to_idx = vocab["token_to_idx"]
idx_to_token = {int(idx): token for token, idx in token_to_idx.items()}
```

**What happens:**
- Loads vocabulary mapping: `{"hello": 42, "world": 17, ...}`
- Creates reverse mapping: `{42: "hello", 17: "world", ...}`
- Extracts special token indices: `<PAD>`, `<START>`, `<END>`

**Why:** The model works with token IDs (integers), not words. Vocabulary converts between text and IDs.

#### 1.2 Load Datasets (lines 192-199)

```python
train_set = EurDatasetLocal(args.data_root, split="train")
test_set = EurDatasetLocal(args.data_root, split="test")
```

**What happens:**
- `EurDatasetLocal` loads `train_data.pkl` and `test_data.pkl`
- Each sample is a list of token IDs: `[1, 42, 17, 2]` (START, hello, world, END)

**Code in `fl_data.py`:**
```python
class EurDatasetLocal(Dataset):
    def __getitem__(self, index):
        return self.data[index]  # Returns list[int] token IDs
```

#### 1.3 Partition Data Across Clients (lines 201-227)

**IID Partitioning (`partition_iid`):**
```python
def partition_iid(num_samples, num_clients, seed=0):
    idx = np.arange(num_samples)  # [0, 1, 2, ..., N-1]
    rng.shuffle(idx)              # Random shuffle
    splits = np.array_split(idx, num_clients)  # Split into N equal parts
    return [s.tolist() for s in splits]
```

**Example with 1000 samples, 4 clients:**
- Client 0: samples [245, 892, 103, ...] (random 250 samples)
- Client 1: samples [67, 334, 901, ...] (random 250 samples)
- Client 2: samples [512, 23, 789, ...] (random 250 samples)
- Client 3: samples [156, 445, 678, ...] (random 250 samples)

**Non-IID Partitioning (`partition_by_length_mild`):**
```python
def partition_by_length_mild(dataset, num_clients, seed=0):
    lengths = np.array([len(dataset[i]) for i in range(len(dataset))])
    idx = np.argsort(lengths)  # Sort by sentence length
    
    # Interleave: client 0 gets idx[0], client 1 gets idx[1], ...
    client_lists = [[] for _ in range(num_clients)]
    for j, i in enumerate(idx):
        client_lists[j % num_clients].append(int(i))
    return client_lists
```

**Example:**
- Short sentences → Client 0, Client 1, Client 2, Client 3, Client 0, ...
- Medium sentences → Client 1, Client 2, Client 3, Client 0, Client 1, ...
- Long sentences → Client 2, Client 3, Client 0, Client 1, Client 2, ...

**Why non-IID?** Real-world federated scenarios often have data heterogeneity (different clients have different data distributions).

**Create DataLoaders:**
```python
for cid in range(args.num_clients):
    subset = Subset(train_set, client_indices[cid])
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, ...)
    client_loaders.append(loader)
```

Each client gets its own DataLoader that provides batches of its local data.

#### 1.4 Initialize Global Model (lines 229-242)

```python
global_model = DeepSC(
    args.num_layers,      # 4 transformer layers
    num_vocab, num_vocab, num_vocab, num_vocab,  # vocab sizes
    args.d_model,         # 128-dimensional embeddings
    args.num_heads,       # 8 attention heads
    args.dff,             # 512 feedforward dimension
    args.dropout          # 0.1 dropout rate
).to(device)

initNetParams(global_model)  # Xavier initialization
```

**What happens:**
- Creates DeepSC model with random weights
- All clients will start from this same global model in round 1

---

### 2. Federated Training Loop (`fl_train.py` lines 246-288)

#### 2.1 Client Selection (lines 255-260)

```python
selected = np.random.choice(
    args.num_clients,                    # Total clients: 20
    size=min(args.clients_per_round, args.num_clients),  # Select 10
    replace=False                         # Without replacement
)
```

**Example:** From 20 clients, randomly select 10: `[0, 3, 7, 12, 15, 2, 9, 18, 5, 11]`

**Why not all clients?**
- Reduces communication overhead
- Faster rounds
- Still provides good model updates

#### 2.2 Local Training (`client_update` function, lines 57-118)

**Step-by-step for one client:**

**A. Copy Global Model:**
```python
model = copy.deepcopy(global_model).to(device)
```
Each client gets an independent copy to avoid modifying the global model.

**B. Create Local Optimizer:**
```python
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, ...)
```
Each client has its own optimizer state.

**C. Local Epochs Loop:**
```python
for local_ep in range(args.local_epochs):  # Default: 1 epoch
    # Sample channel noise for this epoch
    n_var = np.random.uniform(
        SNR_to_noise(args.snr_train_low),   # Convert SNR 5dB to noise variance
        SNR_to_noise(args.snr_train_high),  # Convert SNR 10dB to noise variance
    )
    
    # Train on batches
    for batch_idx, sents in enumerate(client_loader):
        loss = train_step(model, src=sents, trg=sents, n_var=n_var, ...)
        # Backpropagation updates model weights
```

**What `train_step` does (`utils.py` lines 252-301):**

1. **Forward Pass:**
   ```
   Input tokens → Encoder → Channel Encoder → Noisy Channel → 
   Channel Decoder → Decoder → Output predictions
   ```

2. **Channel Simulation:**
   ```python
   if channel == 'AWGN':
       Rx_sig = channels.AWGN(Tx_sig, n_var)  # Add Gaussian noise
   elif channel == 'Rayleigh':
       Rx_sig = channels.Rayleigh(Tx_sig, n_var)  # Fading + noise
   ```

3. **Loss Computation:**
   ```python
   loss = CrossEntropyLoss(predictions, target_tokens)
   loss.backward()  # Compute gradients
   optimizer.step()  # Update weights
   ```

**D. Return Updated Weights:**
```python
return model.state_dict()  # Dictionary of all model parameters
```

**Example state_dict:**
```python
{
    'encoder.embedding.weight': tensor([[...], [...], ...]),
    'encoder.enc_layers.0.mha.wq.weight': tensor([[...], ...]),
    'channel_encoder.0.weight': tensor([[...], ...]),
    ...
}
```

#### 2.3 Aggregation (`fedavg` function, lines 41-54)

**FedAvg (Federated Averaging) Algorithm:**

```python
def fedavg(global_model, client_states, client_sizes):
    total = sum(client_sizes)  # Total samples across selected clients
    
    for param_name in global_model.state_dict().keys():
        # Weighted average: larger datasets have more influence
        new_param = sum(
            client_states[i][param_name] * (client_sizes[i] / total)
            for i in range(len(client_states))
        )
        global_model.state_dict()[param_name] = new_param
```

**Mathematical Formula:**
```
W_global = Σ (n_i / N) * W_i

Where:
- W_global: New global model weights
- W_i: Weights from client i
- n_i: Number of samples at client i
- N: Total samples across all selected clients (Σ n_i)
```

**Example:**
- Client 0: 100 samples, weights W₀
- Client 1: 200 samples, weights W₁
- Client 2: 150 samples, weights W₂
- Total: 450 samples

```
W_global = (100/450) * W₀ + (200/450) * W₁ + (150/450) * W₂
```

**Why weighted?** Clients with more data contribute more to the global model.

#### 2.4 Checkpointing (lines 282-288)

```python
if r % args.save_every == 0:  # Every 10 rounds
    torch.save(global_model.state_dict(), f"fed_deepsc_Rayleigh_round010.pth")
```

Saves model weights periodically for:
- Resuming training
- Evaluation
- Analysis

---

### 3. DeepSC Model Architecture (`models/transceiver.py`)

#### 3.1 Encoder (lines 185-206)

```python
class Encoder(nn.Module):
    def __init__(self, num_layers, src_vocab_size, max_len, d_model, ...):
        self.embedding = nn.Embedding(src_vocab_size, d_model)  # Word → Vector
        self.pos_encoding = PositionalEncoding(...)  # Add position info
        self.enc_layers = nn.ModuleList([EncoderLayer(...) for _ in range(num_layers)])
```

**Forward Pass:**
1. **Embedding:** `[1, 42, 17, 2]` → `[[0.1, 0.3, ...], [0.5, 0.2, ...], ...]` (128-dim vectors)
2. **Positional Encoding:** Add position information
3. **Transformer Layers:** Self-attention + feedforward (4 layers)

**What self-attention does:** Each word attends to all words in the sentence to understand context.

#### 3.2 Channel Encoder (lines 270-273)

```python
self.channel_encoder = nn.Sequential(
    nn.Linear(d_model, 256),    # 128 → 256
    nn.ReLU(),
    nn.Linear(256, 16)          # 256 → 16 (compressed for transmission)
)
```

**Purpose:** Compresses semantic representation for efficient transmission over noisy channel.

#### 3.3 Channel Simulation (`utils.py` lines 169-200)

**AWGN Channel:**
```python
def AWGN(self, Tx_sig, n_var):
    noise = torch.normal(0, n_var, size=Tx_sig.shape)
    Rx_sig = Tx_sig + noise
    return Rx_sig
```
Simply adds Gaussian noise.

**Rayleigh Channel:**
```python
def Rayleigh(self, Tx_sig, n_var):
    # Simulate fading (signal strength varies)
    H = create_fading_matrix()  # Random complex channel matrix
    Tx_sig = Tx_sig @ H         # Apply fading
    Rx_sig = AWGN(Tx_sig, n_var)  # Add noise
    Rx_sig = Rx_sig @ inv(H)    # Channel estimation (try to undo fading)
    return Rx_sig
```
Models wireless fading (signal strength fluctuates).

#### 3.4 Channel Decoder (lines 232-252)

```python
class ChannelDecoder(nn.Module):
    def forward(self, x):
        x1 = self.linear1(x)      # 16 → d_model
        x2 = F.relu(x1)
        x3 = self.linear2(x2)     # d_model → 512
        x4 = F.relu(x3)
        x5 = self.linear3(x4)     # 512 → d_model
        output = layernorm(x1 + x5)  # Residual connection
        return output
```

**Purpose:** Recovers semantic representation from noisy received signal.

#### 3.5 Decoder (lines 210-229)

Similar to encoder but with:
- **Self-attention:** Attends to previously generated words
- **Cross-attention:** Attends to encoder output (semantic representation)
- **Feedforward:** Processes information

**Purpose:** Reconstructs sentence from semantic representation.

---

## Key Concepts Explained

### 1. Why Copy the Global Model?

```python
model = copy.deepcopy(global_model)
```

**Reason:** If we used `model = global_model`, all clients would share the same object. When one client updates weights, it would affect others. Deep copy ensures each client has an independent model.

### 2. Why Sample Channel Noise Per Epoch?

```python
for local_ep in range(args.local_epochs):
    n_var = np.random.uniform(...)  # New noise for each epoch
```

**Reason:** DeepSC needs to be robust to various noise levels. Sampling different noise each epoch improves generalization.

### 3. IID vs. Non-IID Partitioning

**IID (Independent and Identically Distributed):**
- Each client has similar data distribution
- Easier to train, converges faster
- Unrealistic in practice

**Non-IID:**
- Clients have different data distributions
- More realistic (e.g., different users have different writing styles)
- Harder to train, may need more rounds

### 4. FedAvg vs. Simple Average

**Simple Average:**
```
W = (W₁ + W₂ + W₃) / 3
```

**FedAvg (Weighted Average):**
```
W = (n₁/N) * W₁ + (n₂/N) * W₂ + (n₃/N) * W₃
```

**Why FedAvg?** Clients with more data should have more influence on the global model.

### 5. Client Selection Strategy

**Current Implementation:** Random selection
```python
selected = np.random.choice(num_clients, size=clients_per_round, replace=False)
```

**Alternatives (not implemented):**
- **Round-robin:** Select clients in order
- **Stratified:** Ensure diverse client selection
- **Quality-based:** Select clients with better data quality

### 6. Local Epochs vs. Global Rounds

- **Local Epochs (`--local_epochs`):** How many times each client trains on its local data per round
- **Global Rounds (`--rounds`):** How many federated learning rounds to perform

**Trade-off:**
- More local epochs → Better local optimization, but slower rounds
- More rounds → Better global model, but more communication

---

## Running the Code

### Basic Command

```bash
python fl_train.py --data_root data
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_root` | **(required)** | Path to folder containing `europarl/` |
| `--rounds` | 50 | Number of federated learning rounds |
| `--num_clients` | 20 | Total number of clients |
| `--clients_per_round` | 10 | Clients selected per round |
| `--local_epochs` | 1 | Local training epochs per client |
| `--batch_size` | 128 | Batch size for training |
| `--lr` | 1e-4 | Learning rate |
| `--channel` | Rayleigh | Channel type: AWGN, Rayleigh, Rician |
| `--partition` | iid | Data partition: iid, length_mild |
| `--save_dir` | checkpoints_fed | Directory to save checkpoints |

### Example Commands

**Quick test (few rounds, small model):**
```bash
python fl_train.py --data_root data --rounds 5 --num_clients 5 --clients_per_round 3
```

**Full training:**
```bash
python fl_train.py --data_root data --rounds 100 --num_clients 20 --clients_per_round 10 --batch_size 128 --lr 1e-4
```

**Non-IID experiment:**
```bash
python fl_train.py --data_root data --partition length_mild --rounds 50
```

**Different channel:**
```bash
python fl_train.py --data_root data --channel AWGN --rounds 50
```

### Output Files

- `checkpoints_fed/fed_deepsc_Rayleigh_round010.pth` - Checkpoint every 10 rounds
- `checkpoints_fed/fed_deepsc_Rayleigh_final.pth` - Final model

### Monitoring Training

The script prints:
- `[Setup]` - Initialization information
- `[Server] Federated Round X/Y` - Round progress
- `[Server] Selected clients: [...]` - Which clients are training
- `[Client] Local training started` - Client training progress
- `batch X/Y | loss=...` - Training loss per batch
- `[Server] Aggregating client models` - Aggregation step
- `[Server] Checkpoint saved` - Model saved

---

## Summary

This codebase implements **Federated Learning for DeepSC**, a semantic communication system. The workflow:

1. **Initialize:** Load data, partition across clients, create global model
2. **For each round:**
   - Select subset of clients
   - Each client trains locally on its data
   - Server aggregates updates using FedAvg
   - Save checkpoint periodically
3. **Finalize:** Save final global model

**Key Innovation:** Combines federated learning (privacy-preserving distributed training) with semantic communication (efficient, robust text transmission over noisy channels).

**Benefits:**
- Privacy: Text data stays on clients
- Efficiency: Only model updates are communicated
- Robustness: Model learns from diverse distributed data
- Realism: Handles non-IID data distributions

---

## Further Reading

- **FedAvg Paper:** "Communication-Efficient Learning of Deep Networks from Decentralized Data" (McMahan et al., 2017)
- **DeepSC Paper:** "Deep Learning Enabled Semantic Communication Systems" (Xie et al., 2021)
- **Federated Learning Survey:** "Federated Learning: Challenges, Methods, and Future Directions" (Li et al., 2020)

---

*Documentation generated for DeepSC Federated Learning implementation*
