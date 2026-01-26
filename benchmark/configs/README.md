# Benchmark Configuration Guide

This directory contains configuration templates for running person re-identification benchmarks. All configs use YAML format and support auto-detection of dataset-specific parameters.

## Quick Start

```bash
# Preview a configuration
reid config preview benchmark/configs/minimal_template.yaml

# Validate a configuration
reid config validate benchmark/configs/advanced_models.yaml

# Run a benchmark
reid benchmark run --config benchmark/configs/minimal_template.yaml
```

## Available Templates

### 1. `minimal_template.yaml` ⭐ START HERE
**Purpose:** Starting point for creating custom configurations.

**What it contains:**
- Basic structure with detailed comments
- Simple model examples (OSNet, CLIP)
- Commented advanced model examples (CLIP-ReID)
- Minimal configuration for quick experiments

**Use case:** Copy and modify for your experiments

```bash
cp benchmark/configs/minimal_template.yaml my_experiment.yaml
# Edit my_experiment.yaml
python run_benchmark.py --config my_experiment.yaml
```

**Estimated runtime:** ~10-15 minutes (2 models on Market-1501)

---

### 2. `advanced_models.yaml`
**Purpose:** Test state-of-the-art models with various architectures.

**What it runs:**
- **Dataset:** Market-1501 (focused comparison)
- **Models:**
  - CLIP-ReID (2 variants: baseline + SIE+OLP)
  - DINOv2 (3 variants: Base, Large, Giant)
  - SigLIP2 (3 variants: Base 256px, Base 384px, SO400M)
  - PE-Spatial (Dense prediction model)
- **Use case:** Comparing advanced architectures, ablation studies

**Estimated runtime:** Several hours (9 models on 1 dataset)

```bash
python run_benchmark.py --config benchmark/configs/advanced_models.yaml
```

---

### 3. `pespatial.yaml`
**Purpose:** Benchmark PE-Spatial models for person re-identification.

**What it runs:**
- **Dataset:** Market-1501
- **Models:**
  - PE-Spatial-G14-448 (Giant 14, 448×448 resolution)
- **Use case:** Testing PE-Spatial dense prediction models

**Estimated runtime:** ~1-2 hours (1 model on 1 dataset)

```bash
python run_benchmark.py --config-name pespatial
```

---

### 4. `full_evaluation.yaml` ⭐ COMPREHENSIVE
**Purpose:** Complete evaluation - Cross-domain + Zero-shot on all datasets

**What it runs:**
- **Datasets:** 11 datasets (Market-1501, DukeMTMC, CUHK03, MSMT17, GRID, iLIDS, CelebReID, PKU, LASTID, iLIDS-VID, IUST-ReID)
  - Note: G2A excluded (extremely large: 533K queries × 1.9M gallery)
- **Cross-domain models:** CLIP-ReID (2 variants), TransReID trained on MSMT17
- **Zero-shot models:** OSNet variants, CLIP variants, DINOv2, SigLIP2, PE-Core
- **Use case:** Comprehensive benchmark across all available models and datasets

**Estimated runtime:** 12-24 hours (150+ model-dataset combinations)

```bash
python run_benchmark.py --config benchmark/configs/full_evaluation.yaml
```

**Features:**
- Tests generalization of MSMT17-trained models across all domains
- Evaluates zero-shot performance on diverse datasets
- Comprehensive comparison of all model families
- Includes both standard and challenging datasets
- Incremental results saving (results saved after each model+dataset combination)

---

## Configuration Structure

### Basic Configuration

```yaml
# GPU to use (0, 1, 2, etc.)
gpu_id: 0

# Dataset root directory
data:
  root: "reid-data"

# Datasets to benchmark
datasets:
  - market1501
  - dukemtmcreid

# Models to benchmark
models:
  - type: "osnet"
    name: "osnet_x1_0"
    pretrained_path: null

# Output configuration
output:
  results_dir: "."
  csv_filename: "benchmark_results.csv"
```

### Auto-Detection Features

The following parameters are **automatically detected** from the dataset registry and do NOT need to be specified in configs:

- `num_classes` - Number of training identities
- `camera_num` - Number of cameras in the dataset
- `view_num` - Number of viewpoints (usually 1)

**Old way (redundant):**
```yaml
- type: "clipreid"
  name: "ViT-B-16"
  camera_num: 6        # ❌ Not needed
  num_classes: 751     # ❌ Not needed
  view_num: 1
```

**New way (simplified):**
```yaml
- type: "clipreid"
  name: "ViT-B-16"
  view_num: 1          # ✅ Only specify architecture params
  stride_size: [16, 16]
  sie_camera: false
```

---

## Model Configuration Examples

### Simple Models (Minimal Configuration)

```yaml
# OSNet - Lightweight CNN
- type: "osnet"
  name: "osnet_x1_0"
  pretrained_path: null  # Auto-downloads if not found

# CLIP - Zero-shot model
- type: "clip"
  name: "ViT-B/32"

# PE-Core - Large pretrained model
- type: "pecore"
  name: "PE-Core-L14-336"

# PE-Spatial - Dense prediction model
- type: "pespatial"
  name: "PE-Spatial-G14-448"
```

### Advanced Models (More Parameters)

```yaml
# CLIP-ReID - Fine-tuned CLIP
- type: "clipreid"
  name: "ViT-B-16"
  pretrained_path: null
  view_num: 1
  stride_size: [16, 16]      # [12, 12] for overlapping patches
  input_size: [256, 128]
  sie_camera: false          # true to use camera ID
  sie_coe: 1.0

# TransReID - Transformer-based
- type: "transreid"
  name: "vit_base_patch16_224_TransReID"
  pretrained_path: null
  view_num: 1
  stride_size: [16, 16]
  input_size: [256, 128]
  sie_camera: true           # Use side information embedding
  sie_view: false
  sie_coe: 3.0
  jpm: false                 # Jigsaw Patch Module
  drop_path: 0.1
  drop_out: 0.0
  att_drop_rate: 0.0
```

### Model Variants Guide

**CLIP-ReID:**
- `sie_camera=false, stride_size=[16,16]` → Baseline variant
- `sie_camera=true, stride_size=[16,16]` → With Side Information Embedding
- `sie_camera=true, stride_size=[12,12]` → **Best performance** (SIE + Overlapping Local Patches)

**TransReID:**
- `sie_camera=false, jpm=false` → Baseline ViT
- `sie_camera=true, jpm=false` → With SIE (recommended)
- `sie_camera=true, jpm=true` → Full TransReID (best accuracy)

---

## CLI Commands

### Configuration Management

```bash
# List all available configs
reid config list

# View a config file
reid config show minimal_template
reid config show advanced_models

# Validate a config
reid config validate my_config.yaml
reid config validate my_config.yaml --strict  # Stricter validation

# Preview what will run
reid config preview my_config.yaml

# List available templates
reid config templates
```

### Interactive Config Builder

Create a configuration interactively with guided prompts:

```bash
# Interactive wizard
reid config create

# Save to specific file
reid config create --output my_experiment.yaml
```

The wizard will guide you through:
1. GPU selection
2. Dataset selection (shows download status)
3. Model selection (with descriptions)
4. Model-specific parameters (only for complex models)
5. Output configuration

---

## Cross-Domain Configuration

For domain generalization experiments:

```yaml
# Models trained on source domain, evaluated on targets
models:
  - type: "clipreid"
    name: "ViT-B-16"
    source_domain: "msmt17"  # Use MSMT17 pretrained weights
    pretrained_path: null
    stride_size: [16, 16]
    sie_camera: false
    # ... other params

# Evaluate on all target datasets
datasets:
  - msmt17        # In-domain baseline
  - market1501    # Cross-domain
  - dukemtmcreid  # Cross-domain
  - cuhk03        # Cross-domain
```

**Important notes:**
- In cross-domain mode, the classifier head is NOT loaded (feature extraction only)
- Camera/class counts are taken from the TARGET dataset, not source
- Good generalization typically shows <10% drop from in-domain to cross-domain

---

## Available Datasets

Check which datasets are available:

```bash
# List all registered datasets
reid dataset list

# Check which are downloaded
reid dataset check

# Get dataset details
reid dataset info market1501
```

Common datasets:
- `market1501` - 32,668 images, 1,501 IDs, 6 cameras
- `dukemtmcreid` - 36,411 images, 1,404 IDs, 8 cameras
- `cuhk03` - 28,192 images, 1,467 IDs
- `msmt17` - 126,441 images, 4,101 IDs, 15 cameras (largest)
- `viper` - Small dataset for quick tests
- `grid` - Small dataset for quick tests

---

## Available Models

Check which model types are supported:

```bash
# List all model types
reid model list

# Get model details
reid model info clipreid
```

Supported model types:
- `osnet` - Lightweight CNN baseline
- `clip` - Zero-shot vision-language model
- `clipreid` - Fine-tuned CLIP for person ReID
- `transreid` - Transformer-based ReID
- `pecore` - Large-scale pretrained vision model
- `pespatial` - Dense prediction model with spatial understanding
- `dinov2` - Self-supervised ViT
- `siglip2` - Sigmoid loss CLIP variant

Note: `dinov3` is not yet released (wrapper falls back to dinov2)

---

## Tips and Best Practices

### 1. Start Small
Begin with `minimal_template.yaml` to verify everything works before running large benchmarks.

### 2. Use Preview
Always preview your config before running to catch issues early:
```bash
reid config preview my_config.yaml
```

### 3. Validate First
Run validation to check for errors:
```bash
reid config validate my_config.yaml --strict
```

### 4. GPU Management
If you have multiple GPUs, set `gpu_id` appropriately:
```yaml
gpu_id: 0  # Use first GPU
gpu_id: 1  # Use second GPU
```

Check available GPUs:
```bash
reid info  # Shows system info including GPUs
```

### 5. Model Weights
Pretrained weights are automatically downloaded when `pretrained_path: null`. They're cached in `pretrained_models/` directory.

### 6. Output Organization
Organize results by experiment:
```yaml
output:
  results_dir: "results/experiment_1"
  csv_filename: "ablation_study.csv"
```

### 7. Incremental Testing
Test models incrementally instead of all at once:
1. Start with 1 dataset, 1 model
2. Add more models
3. Add more datasets
4. Run full benchmark

### 8. Incremental Results Saving
Results are now saved after **each model+dataset combination** completes, so you never lose progress if the benchmark crashes or is interrupted.

---

## Troubleshooting

### Config won't validate
```bash
# Run validation to see specific errors
reid config validate my_config.yaml --strict
```

Common issues:
- Missing required keys (`datasets`, `models`)
- Wrong data types (e.g., `gpu_id` must be integer)
- Unknown dataset or model type
- Dataset not downloaded

### Preview shows wrong parameters
Auto-detection happens at runtime, not during preview. The preview shows what's in the YAML file.

### Model-specific parameters confusing
Use the interactive builder for guidance:
```bash
reid config create
```

It will prompt for only the necessary parameters for each model type.

---

## Migration from Old Configs

If you have old config files with redundant parameters, here's how to clean them up:

**Before (old style):**
```yaml
models:
  - type: "clipreid"
    name: "ViT-B-16"
    camera_num: 6        # Remove - auto-detected
    num_classes: 751     # Remove - auto-detected
    view_num: 1
    stride_size: [16, 16]
    sie_camera: false
```

**After (new style):**
```yaml
models:
  - type: "clipreid"
    name: "ViT-B-16"
    view_num: 1
    stride_size: [16, 16]
    sie_camera: false
```

The validation command will warn you about redundant parameters:
```bash
reid config validate old_config.yaml
# INFO: Model #1: 'camera_num' is auto-detected from dataset (can be omitted)
# INFO: Model #2: 'num_classes' is auto-detected from dataset (can be omitted)
```

---

## Getting Help

```bash
# General help
reid --help

# Config-specific help
reid config --help

# Command-specific help
reid config create --help
reid config validate --help
reid config preview --help
```

For issues or questions, check the main README or open an issue on GitHub.
