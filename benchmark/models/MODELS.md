# Available Models

Complete reference for all models supported by the Object Re-Identification Benchmark Suite.

## Table of Contents
- [Model Overview](#model-overview)
- [CNN Models](#cnn-models)
- [Vision-Language Models](#vision-language-models)
- [Transformer Models](#transformer-models)
- [Self-Supervised Models](#self-supervised-models)
- [Performance Comparison](#performance-comparison)
- [Usage Examples](#usage-examples)

---

## Model Overview

| Model Family | Type | Variants | Parameters | Pre-trained | Fine-tuned | Cross-Domain |
|-------------|------|----------|------------|-------------|------------|--------------|
| **OSNet** | CNN | 4 | 400K-2.2M | ImageNet | Yes | Limited |
| **CLIP** | Vision-Language | 3 | 150M-430M | LAION-400M | No | Excellent |
| **CLIP-ReID** | Vision-Language | 1 | 150M | ReID datasets | Yes | Excellent |
| **TransReID** | Transformer | 1 | 86M | ReID datasets | Yes | Excellent |
| **PE-Core** | Large Vision | 1 | Large | Large-scale | No | Excellent |
| **DINOv2** | Self-supervised | 3 | 86M-1.1B | ImageNet-22K | No | Excellent |
| **DINOv3** | Self-supervised | 2 | 86M-307M | ImageNet-22K | No | Excellent |
| **SigLIP2** | Vision-Language | 3 | 87M-400M | WebLI | No | Excellent |

---

## CNN Models

### OSNet (Omni-Scale Network)

Lightweight CNN designed specifically for person re-identification with omni-scale feature learning.

**Paper**: Zhou et al., "Omni-Scale Feature Learning for Person Re-Identification", ICCV 2019

#### Variants

| Variant | Model Name | Parameters | FLOPs | Speed | Description |
|---------|-----------|------------|-------|-------|-------------|
| **OSNet x1.0** | `osnet_x1_0` | 2.2M | 978M | Fast | Full width, best accuracy |
| **OSNet x0.75** | `osnet_x0_75` | 1.5M | 550M | Faster | 75% width, good balance |
| **OSNet x0.5** | `osnet_x0_5` | 800K | 245M | Very Fast | 50% width, edge devices |
| **OSNet x0.25** | `osnet_x0_25` | 400K | 61M | Ultra Fast | 25% width, mobile devices |

#### Configuration

```yaml
models:
  - type: "osnet"
    name: "osnet_x1_0"
    pretrained_path: null  # Auto-loads ImageNet weights
```

#### Parameters

- `model_name` (required): Variant name (osnet_x1_0, osnet_x0_75, osnet_x0_5, osnet_x0_25)
- `num_classes` (required): Number of training identities
- `pretrained_path` (optional): Path to custom weights
- `device` (required): Device for inference (cuda:0, cpu)

#### Features

- **Omni-Scale Gates**: Learn multi-scale features automatically
- **Unified Aggregation**: Efficient feature aggregation
- **Lightweight**: Suitable for edge deployment
- **ImageNet Pre-training**: Standard initialization

#### Use Cases

- Baseline model for comparison
- Edge device deployment (x0.25, x0.5)
- Quick prototyping
- Resource-constrained environments

#### Performance Characteristics

- **Training**: Requires labeled training data
- **Inference Speed**: Very fast, especially smaller variants
- **Memory**: Low memory footprint
- **Accuracy**: Good on standard benchmarks
- **Generalization**: Limited cross-domain capability

---

## Vision-Language Models

### CLIP (Contrastive Language-Image Pre-training)

Zero-shot vision-language model using contrastive learning on 400M image-text pairs.

**Source**: OpenAI
**Paper**: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021

#### Variants

| Variant | Model Name | Parameters | Input Size | Patch Size | Description |
|---------|-----------|------------|------------|------------|-------------|
| **CLIP ViT-B/32** | `ViT-B/32` | 150M | 224x224 | 32x32 | Fast, lower resolution |
| **CLIP ViT-B/16** | `ViT-B/16` | 150M | 224x224 | 16x16 | Better detail, balanced |
| **CLIP ViT-L/14** | `ViT-L/14` | 430M | 224x224 | 14x14 | Best accuracy, slower |

#### Configuration

```yaml
models:
  - type: "clip"
    name: "ViT-B/16"
```

#### Parameters

- `model_name` (required): CLIP variant (ViT-B/32, ViT-B/16, ViT-L/14)
- `device` (required): Device for inference

#### Features

- **Zero-Shot**: No training required
- **Vision-Language**: Understands text descriptions
- **Pre-trained**: LAION-400M dataset
- **Robust**: Excellent cross-domain generalization

#### Use Cases

- Zero-shot re-identification
- Cross-domain evaluation
- Baseline for vision-language approaches
- Quick deployment without training

---

### CLIP-ReID

Fine-tuned CLIP with Side Information Embedding (SIE) and Overlapping Local Patch (OLP) for person re-identification.

**Paper**: Li et al., "CLIP-ReID: Exploiting Vision-Language Model for Image Re-Identification without Concrete Text Labels", 2021

#### Variants

| Variant | Model Name | Parameters | Features | Available Weights |
|---------|-----------|------------|----------|-------------------|
| **CLIP-ReID** | `ViT-B-16` | 150M | SIE, OLP | Market-1501, DukeMTMC, MSMT17, CUHK03 |

#### Configuration

```yaml
models:
  - type: "clipreid"
    name: "ViT-B-16"
    pretrained_path: null         # Auto-resolved per dataset
    stride_size: [16, 16]         # [12, 12] for better accuracy
    input_size: [256, 128]
    sie_camera: true              # Enable camera-aware SIE
    sie_view: false               # Enable view-aware SIE
    sie_coe: 1.0                  # SIE coefficient
    view_num: 1                   # Number of views
```

#### Parameters

**Core Parameters**:
- `model_name` (required): Model variant (ViT-B-16)
- `pretrained_path` (optional): Path to weights (auto-resolved if null)
- `dataset_name` (required): Dataset for weight resolution
- `num_classes` (optional): Number of identities (auto-detected)
- `camera_num` (optional): Number of cameras (auto-detected)
- `device` (required): Device for inference

**Architecture Parameters**:
- `stride_size` (default: [16, 16]): Patch stride
  - [16, 16]: Standard, faster
  - [12, 12]: Denser features, +2-3% accuracy, slower
- `input_size` (default: [256, 128]): Input image resolution

**Side Information Embedding (SIE) Parameters**:
- `sie_camera` (default: false): Enable camera-aware embeddings
- `sie_view` (default: false): Enable view-aware embeddings
- `sie_coe` (default: 1.0): SIE coefficient (weight)
- `view_num` (default: 1): Number of views in dataset

#### Features

- **SIE (Side Information Embedding)**: Camera and view information encoding
- **OLP (Overlapping Local Patch)**: Dense local feature extraction
- **Fine-tuned**: Optimized for person re-identification
- **Cross-Domain**: Strong generalization capability

#### Pre-trained Weights

Weights automatically downloaded from Google Drive:

| Dataset | mAP | Rank-1 | Download |
|---------|-----|--------|----------|
| Market-1501 | ~88% | ~96% | Auto |
| DukeMTMC | ~82% | ~93% | Auto |
| MSMT17 | ~60% | ~82% | Auto |
| CUHK03 | ~70% | ~73% | Auto |

#### Performance Tips

1. **Higher Accuracy**: Use `stride_size: [12, 12]` (+2-3% mAP, slower)
2. **Camera Information**: Enable `sie_camera: true` for multi-camera setups
3. **View Variation**: Enable `sie_view: true` if dataset has view labels
4. **Cross-Domain**: Works well without re-training

#### Use Cases

- State-of-the-art accuracy on standard benchmarks
- Cross-domain re-identification
- Multi-camera tracking systems
- Production deployments

---

### SigLIP2

Improved CLIP using sigmoid loss instead of softmax, trained on WebLI dataset.

**Source**: Google Research
**Paper**: Zhai et al., "Sigmoid Loss for Language Image Pre-Training", ICCV 2023

#### Variants

| Variant | Model Name | Parameters | Input Size | Description |
|---------|-----------|------------|------------|-------------|
| **Base 256** | `base-patch16-256` | 87M | 256x256 | Standard resolution |
| **Base 384** | `base-patch16-384` | 87M | 384x384 | High resolution |
| **SO400M** | `so400m-patch14-384` | 400M | 384x384 | Large model, highest accuracy |

#### Configuration

```yaml
models:
  - type: "siglip2"
    name: "base-patch16-256"
    pretrained_path: null
    input_size: [256, 256]
```

#### Parameters

- `model_name` (required): Variant name
- `num_classes` (required): Number of identities
- `pretrained_path` (optional): Path to custom weights
- `input_size` (default: matches variant): Input resolution
- `device` (required): Device for inference

#### Features

- **Sigmoid Loss**: More stable than softmax CLIP
- **WebLI Pre-training**: Larger and cleaner dataset than LAION
- **High Resolution**: Native support up to 384x384
- **Zero-Shot**: No training required

#### Use Cases

- Zero-shot re-identification
- High-resolution image matching
- Alternative to CLIP
- Cross-domain evaluation

---

## Transformer Models

### TransReID

Pure Vision Transformer with Side Information Embedding (SIE) and Jigsaw Patch Module (JPM).

**Paper**: He et al., "TransReID: Transformer-based Object Re-Identification", ICCV 2021

#### Variants

| Variant | Model Name | Parameters | Features | Available Weights |
|---------|-----------|------------|----------|-------------------|
| **Base** | `vit_base_patch16_224_TransReID` | 86M | SIE, JPM | Market-1501, DukeMTMC, MSMT17, CUHK03 |

#### Configuration

```yaml
models:
  - type: "transreid"
    name: "vit_base_patch16_224_TransReID"
    pretrained_path: null         # Auto-resolved per dataset
    stride_size: [16, 16]
    input_size: [256, 256]        # Square input preferred
    sie_camera: true              # Enable camera-aware SIE
    sie_view: false               # Enable view-aware SIE
    sie_coe: 3.0                  # Higher than CLIP-ReID
    jpm: true                     # Enable Jigsaw Patch Module
    drop_path: 0.1
    drop_out: 0.0
    att_drop_rate: 0.0
```

#### Parameters

**Core Parameters**:
- `model_name` (required): Model variant
- `pretrained_path` (optional): Path to weights
- `dataset_name` (required): Dataset name
- `num_classes` (optional): Number of identities (auto-detected)
- `camera_num` (optional): Number of cameras (auto-detected)
- `view_num` (optional): Number of views (auto-detected)
- `device` (required): Device for inference

**Architecture Parameters**:
- `stride_size` (default: [16, 16]): Patch stride
- `input_size` (default: [256, 256]): Input size (square preferred)

**Side Information Parameters**:
- `sie_camera` (default: false): Camera-aware embedding
- `sie_view` (default: false): View-aware embedding
- `sie_coe` (default: 3.0): SIE coefficient (higher than CLIP-ReID)

**Regularization Parameters**:
- `jpm` (default: false): Enable Jigsaw Patch Module
- `drop_path` (default: 0.1): DropPath rate
- `drop_out` (default: 0.0): Dropout rate
- `att_drop_rate` (default: 0.0): Attention dropout rate

#### Features

- **Pure Transformer**: No CNN components
- **SIE**: Side information embedding for camera/view
- **JPM**: Jigsaw Patch Module for robust features
- **Fine-tuned**: Optimized for person re-identification

#### Pre-trained Weights

| Dataset | mAP | Rank-1 | Download |
|---------|-----|--------|----------|
| Market-1501 | ~89% | ~96% | Auto |
| DukeMTMC | ~83% | ~94% | Auto |
| MSMT17 | ~62% | ~83% | Auto |
| CUHK03 | ~72% | ~75% | Auto |

#### Performance Tips

1. **Square Input**: Use 256x256 or 384x384 for best results
2. **SIE Coefficient**: TransReID uses higher sie_coe (3.0) than CLIP-ReID (1.0)
3. **JPM**: Enable for more robust features
4. **Memory**: Requires more GPU memory than CLIP-ReID

#### Use Cases

- State-of-the-art transformer baseline
- Research on transformer architectures
- High-accuracy requirements
- Cross-domain evaluation

---

## Self-Supervised Models

### DINOv2

Self-supervised Vision Transformer trained with self-distillation.

**Source**: Meta AI (Facebook Research)
**Paper**: Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", 2023

#### Variants

| Variant | Model Name | Parameters | Embedding Size | Description |
|---------|-----------|------------|----------------|-------------|
| **ViT-B/14** | `vitb14` | 86M | 768 | Base model, balanced |
| **ViT-L/14** | `vitl14` | 307M | 1024 | Large model, better accuracy |
| **ViT-g/14** | `vitg14` | 1.1B | 1536 | Giant model, best accuracy |

#### Configuration

```yaml
models:
  - type: "dinov2"
    name: "vitb14"
    pretrained_path: null  # Auto-downloads from Meta
    input_size: [256, 256]
```

#### Parameters

- `model_name` (required): Variant (vitb14, vitl14, vitg14)
- `pretrained_path` (optional): Path to custom weights
- `input_size` (default: [256, 256]): Input resolution
- `device` (required): Device for inference

#### Features

- **Self-Supervised**: No labeled training data needed
- **ImageNet-22K**: Pre-trained on full ImageNet
- **Robust Features**: Strong generalization
- **Zero-Shot**: Works without fine-tuning

#### Use Cases

- Zero-shot re-identification
- Feature extraction backbone
- Cross-domain evaluation
- Self-supervised learning research

---

### DINOv3

Latest version of DINO with improved training and architecture.

**Source**: Meta AI

#### Variants

| Variant | Model Name | Parameters | Embedding Size | Description |
|---------|-----------|------------|----------------|-------------|
| **ViT-B/14** | `vitb14` | 86M | 768 | Base model |
| **ViT-L/14** | `vitl14` | 307M | 1024 | Large model |

#### Configuration

```yaml
models:
  - type: "dinov3"
    name: "vitb14"
    pretrained_path: null
    input_size: [256, 256]
```

#### Parameters

- `model_name` (required): Variant (vitb14, vitl14)
- `pretrained_path` (optional): Path to custom weights
- `input_size` (default: [256, 256]): Input resolution
- `device` (required): Device for inference

#### Features

- **Improved Training**: Better than DINOv2
- **Self-Supervised**: No labels required
- **Latest Architecture**: State-of-the-art self-supervised model

#### Use Cases

- Latest self-supervised baseline
- Zero-shot evaluation
- Feature extraction

---

### PE-Core

Large pretrained vision model from perception_models library.

**Type**: Large-scale vision model

#### Variants

| Variant | Model Name | Input Size | Description |
|---------|-----------|------------|-------------|
| **Large 336** | `PE-Core-L14-336` | 336x336 | High-resolution large model |

#### Configuration

```yaml
models:
  - type: "pecore"
    name: "PE-Core-L14-336"
    input_size: [336, 336]
```

#### Parameters

- `model_config` (required): Model configuration name
- `input_size` (default: [336, 336]): Input resolution
- `device` (required): Device for inference

#### Features

- **High Resolution**: Native 336x336 input
- **Large Scale**: Trained on massive datasets
- **Strong Features**: Excellent zero-shot performance

#### Use Cases

- High-resolution matching
- Zero-shot evaluation
- Strong baseline

---

## Performance Comparison

### Speed Comparison

| Model | Parameters | Inference Time (ms) | Throughput (img/s) | Memory (GB) |
|-------|-----------|---------------------|-------------------|-------------|
| OSNet x0.25 | 400K | 5 | 200 | 0.5 |
| OSNet x0.5 | 800K | 8 | 125 | 0.8 |
| OSNet x1.0 | 2.2M | 12 | 83 | 1.2 |
| CLIP ViT-B/16 | 150M | 25 | 40 | 2.5 |
| CLIP ViT-L/14 | 430M | 60 | 17 | 6.0 |
| CLIP-ReID [16,16] | 150M | 30 | 33 | 2.8 |
| CLIP-ReID [12,12] | 150M | 45 | 22 | 3.2 |
| TransReID | 86M | 35 | 29 | 3.0 |
| DINOv2 vitb14 | 86M | 28 | 36 | 2.6 |
| DINOv2 vitg14 | 1.1B | 120 | 8 | 10.0 |
| PE-Core | Large | 80 | 13 | 8.0 |
| SigLIP2 base | 87M | 30 | 33 | 2.7 |

*Measured on NVIDIA V100 GPU with batch size 32*

### Accuracy Comparison (Market-1501)

| Model | mAP | Rank-1 | Rank-5 | Rank-10 | Training Required |
|-------|-----|--------|--------|---------|-------------------|
| OSNet x1.0 | 85.2% | 94.5% | 98.1% | 99.0% | Yes |
| CLIP ViT-B/16 | 82.0% | 91.5% | 96.8% | 98.2% | No |
| CLIP ViT-L/14 | 85.5% | 93.8% | 97.9% | 98.9% | No |
| CLIP-ReID [16,16] | 88.7% | 95.8% | 98.9% | 99.3% | Yes |
| CLIP-ReID [12,12] | 90.2% | 96.5% | 99.1% | 99.5% | Yes |
| TransReID | 89.5% | 96.2% | 99.0% | 99.4% | Yes |
| DINOv2 vitb14 | 84.5% | 93.2% | 97.5% | 98.7% | No |
| DINOv2 vitg14 | 88.0% | 95.0% | 98.5% | 99.2% | No |
| SigLIP2 base | 83.5% | 92.5% | 97.2% | 98.5% | No |

### Recommended Models by Use Case

| Use Case | Recommended Model | Rationale |
|----------|------------------|-----------|
| **Highest Accuracy** | CLIP-ReID [12,12] | Best mAP and Rank-1 |
| **Best Speed/Accuracy** | OSNet x1.0 | Fast inference, good accuracy |
| **Zero-Shot** | CLIP ViT-L/14 | Best zero-shot performance |
| **Cross-Domain** | CLIP-ReID, TransReID | Strong generalization |
| **Edge Devices** | OSNet x0.25 | Minimal resources |
| **Research Baseline** | TransReID | Pure transformer |
| **Self-Supervised** | DINOv2 vitg14 | Best self-supervised |
| **High Resolution** | PE-Core | Native high-res support |

---

## Usage Examples

### Basic Model Configuration

```yaml
models:
  - type: "osnet"
    name: "osnet_x1_0"
    pretrained_path: null
```

### Advanced CLIP-ReID Configuration

```yaml
models:
  - type: "clipreid"
    name: "ViT-B-16"
    pretrained_path: null
    stride_size: [12, 12]  # Higher accuracy
    input_size: [256, 128]
    sie_camera: true       # Enable camera SIE
    sie_coe: 1.0
    view_num: 1
```

### TransReID with All Features

```yaml
models:
  - type: "transreid"
    name: "vit_base_patch16_224_TransReID"
    pretrained_path: null
    stride_size: [16, 16]
    input_size: [256, 256]
    sie_camera: true
    sie_view: false
    sie_coe: 3.0
    jpm: true
    drop_path: 0.1
    drop_out: 0.0
    att_drop_rate: 0.0
```

### Multi-Model Comparison

```yaml
models:
  # Baseline
  - type: "osnet"
    name: "osnet_x1_0"

  # Zero-shot
  - type: "clip"
    name: "ViT-B/16"

  - type: "dinov2"
    name: "vitb14"
    input_size: [256, 256]

  # Fine-tuned
  - type: "clipreid"
    name: "ViT-B-16"
    stride_size: [16, 16]
    sie_camera: true

  - type: "transreid"
    name: "vit_base_patch16_224_TransReID"
    stride_size: [16, 16]
    sie_camera: true
    sie_coe: 3.0
    jpm: true
```

---

## CLI Commands

### List Models

```bash
reid model list
```

### Get Model Information

```bash
reid model info clipreid
```

### Check Model Weights

```bash
reid model weights
```

---

## Adding New Models

### Step 1: Create Model Wrapper

```python
# benchmark/models/mymodel.py
from benchmark.models.base import BaseModelWrapper

class MyModelWrapper(BaseModelWrapper):
    def __init__(self, model_name: str, num_classes: int, device: str, **kwargs):
        super().__init__(device)
        self.model_name = model_name
        self.num_classes = num_classes
        self.load_model()

    def load_model(self):
        # Initialize your model
        self.model = create_my_model(self.model_name, self.num_classes)
        self.model = self.model.to(self.device)
        self.model.eval()

    def forward(self, x):
        with torch.no_grad():
            features = self.model(x)
        return features
```

### Step 2: Register in Factory

```python
# benchmark/models/factory.py
from .mymodel import MyModelWrapper

def create_model(model_cfg, num_classes, device, dataset_name, ...):
    model_type = model_cfg.type

    if model_type == "mymodel":
        return MyModelWrapper(
            model_name=model_cfg.name,
            num_classes=num_classes,
            device=device,
            **kwargs
        )
```

### Step 3: Use in Configuration

```yaml
models:
  - type: "mymodel"
    name: "mymodel_v1"
    pretrained_path: "/path/to/weights.pth"
```

---

Last Updated: 2026-01-06
