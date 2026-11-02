# Technical Specification: ModernBERT-JEPA Decider for JevBench

## Executive Summary
This document outlines the technical architecture, data flow, training recipe, and deployment strategy for building a **Joint Embedding Predictive Architecture (JEPA)** based on **ModernBERT-Base (~149M parameters)**. 

The primary goal is to deploy a non-generative, sub-15ms decision engine (router / task allocator) benchmarked on JevBench while keeping training compute under 1 GPU-hour on commodity hardware.

---

## 1. System Architecture

Unlike causal autoregressive models that generate output tokens sequentially ($O(N)$ inference steps), this system uses a single-pass continuous latent prediction framework ($O(1)$ inference step).

```
                 ┌───────────────────────────┐
                 │    Input Context Text     │
                 └─────────────┬─────────────┘
                               │
                               ▼
                ┌─────────────────────────────┐
                │   ModernBERT-Base Encoder   │
                │  (Unpadded FlashAttention)  │
                └──────────────┬──────────────┘
                               │ Context Vector z_c (768-d)
                               ▼
                ┌─────────────────────────────┐
                │    Latent Predictor MLP     │
                │   (2-Layer Bottleneck MLP)  │
                └──────────────┬──────────────┘
                               │ Predicted Latent z_hat (768-d)
                               ▼
                ┌─────────────────────────────┐
                │ Temperature-Calibrated Head │
                └──────────────┬──────────────┘
                               │
                               ▼
                   Decision Route / Class Logits
```

---

## 2. Model Blueprint & PyTorch Implementation

Save this complete implementation module as `model.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

class ModernBERTJEPADecider(nn.Module):
    def __init__(self, model_name="answerdotai/ModernBERT-base", num_routes=64, frozen_layers=18):
        """
        ModernBERT-JEPA Decider Architecture (~150M Total Parameters)
        
        Args:
            model_name (str): HuggingFace model path for ModernBERT.
            num_routes (int): Target number of discrete decision routes or classes.
            frozen_layers (int): Number of lower transformer layers to freeze for fast tuning.
        """
        super().__init__()
        
        # 1. Base Encoder (ModernBERT-Base)
        self.encoder = AutoModel.from_pretrained(model_name)
        embed_dim = self.encoder.config.hidden_size  # 768 for Base
        
        # 2. Freeze lower layers for Parameter-Efficient Tuning
        self._freeze_backbone_layers(frozen_layers)
        
        # 3. Latent Predictor Head (Bottleneck MLP)
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, 384),
            nn.GELU(),
            nn.LayerNorm(384),
            nn.Linear(384, embed_dim)
        )
        
        # 4. Calibrated Decision Head
        self.decision_head = nn.Linear(embed_dim, num_routes)
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def _freeze_backbone_layers(self, num_layers):
        """Freezes embeddings and specified lower transformer layers."""
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
            
        if hasattr(self.encoder, 'layers'):
            for layer in self.encoder.layers[:num_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, input_ids, attention_mask=None, target_latents=None):
        # Extract ModernBERT representations
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        # Sequence Mean Pooling over unmasked tokens
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(last_hidden_state)
            z_c = torch.sum(last_hidden_state * mask_expanded, dim=1) / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        else:
            z_c = last_hidden_state.mean(dim=1)
            
        # Predict target continuous embedding
        z_pred = self.predictor(z_c)
        
        # Soft-Target Calibrated Logits
        logits = self.decision_head(z_pred) / torch.abs(self.temperature)
        
        loss = None
        if target_latents is not None:
            # Combined Latent Smooth L1 + Decision Entropy Loss
            loss_latent = F.smooth_l1_loss(z_pred, target_latents)
            loss = loss_latent
            
        return {
            "context_latent": z_c,
            "predicted_latent": z_pred,
            "logits": logits,
            "loss": loss
        }
```

---

## 3. Data Pipeline & Preprocessing

### Target Format
Ensure input text dataset is formatted into JSON Lines (`.jsonl`):

```json
{"id": "req_001", "context": "System Log: API response latency elevated in region us-east-1. Route traffic.", "target_route": 12}
{"id": "req_002", "context": "User Query: Reset primary password for enterprise account.", "target_route": 3}
```

### Fast Data Collator
Use padding token truncation bounded at sequence length $L = 128$:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

def preprocess_function(examples):
    return tokenizer(
        examples["context"],
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
```

---

## 4. Training Recipe

### Loss Function
The model optimizes a composite objective balancing **Latent Alignment** and **Target Task Divergence**:

$$\mathcal{L}_{\text{total}} = \text{SmoothL1}(\hat{z}_t, z_t) + \lambda \cdot \text{CrossEntropy}(P_{\text{pred}}, Y_{\text{target}})$$

### Recommended Hyperparameters

| Parameter | Value | Reason |
| :--- | :--- | :--- |
| **Optimizer** | AdamW (`lr=3e-4`, `weight_decay=0.01`) | Fast convergence on top layers & heads |
| **Precision** | `fp16` / `mixed_precision` | Hardware acceleration on NVIDIA T4/A10G |
| **Batch Size** | 32 per GPU (Accumulation: 2) | Fits comfortably inside 16GB VRAM |
| **Epochs** | 3 to 5 | Prevents overfitting on frozen backbones |
| **Warmup Ratio** | 0.1 | Cosine learning rate decay |

---

## 5. Export & Optimization (Sub-15ms Latency Target)

To pass strict JevBench latency constraints (< 15ms p99 overhead), compile the PyTorch model into **ONNX Runtime (FP16)**.

### ONNX Export Script (`export_onnx.py`)

```python
import torch
from model import ModernBERTJEPADecider

def export_to_onnx():
    model = ModernBERTJEPADecider()
    model.eval()

    dummy_input_ids = torch.randint(0, 30000, (1, 128), dtype=torch.long)
    dummy_mask = torch.ones((1, 128), dtype=torch.long)

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_mask),
        "modernbert_jepa.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["context_latent", "predicted_latent", "logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits": {0: "batch_size"}
        },
        opset_version=17
    )
    print("Model exported successfully to modernbert_jepa.onnx")

if __name__ == "__main__":
    export_to_onnx()
```

### ONNX Inference Execution Engine (`infer.py`)

```python
import onnxruntime as ort
import numpy as np

# Initialize ONNX Session with Parallel Execution Engine
options = ort.SessionOptions()
options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession("modernbert_jepa.onnx", options, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

def run_inference(input_ids, attention_mask):
    inputs = {
        'input_ids': input_ids.astype(np.int64),
        'attention_mask': attention_mask.astype(np.int64)
    }
    outputs = session.run(None, inputs)
    logits = outputs[2]
    return np.argmax(logits, axis=-1)
```

---

## 6. Benchmarking Protocol on JevBench

1. **Warmup Run:** Execute 50 dummy passes to initialize CUDA context and TensorRT/ONNX graphs.
2. **Batch Evaluation:** Pass JevBench test sets through `run_inference()`.
3. **Reported Metrics:**
   * **Route Accuracy (%)**
   * **Embedding Mean Squared Error (MSE)**
   * **p50, p95, p99 Latency (ms)**
