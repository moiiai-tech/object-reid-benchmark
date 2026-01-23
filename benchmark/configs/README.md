# Benchmark Configuration Guide

This directory contains configuration templates for running person re-identification benchmarks. All configs use YAML format and support auto-detection of dataset-specific parameters.

## Quick Start

```bash
# List available templates
reid config templates

# Preview a configuration
reid config preview benchmark/configs/quick_validation.yaml

# Validate a configuration
reid config validate benchmark/configs/comprehensive_benchmark.yaml

# Run a benchmark
reid benchmark run --config benchmark/configs/quick_validation.yaml
```

## Available Templates

### 1. `quick_validation.yaml`
**Purpose:** Fast smoke test to verify your setup works correctly.

**What it runs:**
- **Dataset:** Market-1501 only
- **Model:** OSNet (lightweight baseline)
- **Use case:** Quick validation, debugging, testing new code changes

**Estimated runtime:** ~5-10 minutes

```bash
python run_benchmark.py --config benchmark/configs/quick_validation.yaml
```

---

### 2. `comprehensive_benchmark.yaml`
**Purpose:** Full evaluation across major datasets with standard models.

**What it runs:**
- **Datasets:** Market-1501, DukeMTMC-reID, CUHK03, MSMT17
- **Models:** OSNet, CLIP, CLIP-ReID, TransReID, PE-Core
- **Use case:** Comprehensive model comparison, paper benchmarks

**Estimated runtime:** Several hours (20 model-dataset combinations)

```bash
python run_benchmark.py --config benchmark/configs/comprehensive_benchmark.yaml
```

---

### 3. `advanced_models.yaml`
**Purpose:** Test state-of-the-art models with various architectures.

**What it runs:**
- **Dataset:** Market-1501 (focused comparison)
- **Models:** CLIP-ReID variants, DINOv2, DINOv3, SigLIP2
- **Use case:** Comparing advanced architectures, ablation studies

**Estimated runtime:** Several hours (9 models on 1 dataset)

```bash
python run_benchmark.py --config benchmark/configs/advanced_models.yaml
```

---

### 4. `cross_domain_test.yaml`
**Purpose:** Evaluate domain generalization without fine-tuning.

**What it runs:**
- **Source domain:** Market-1501 (where model was trained)
- **Target domains:** Market-1501, DukeMTMC-reID, CUHK03, MSMT17
- **Models:** CLIP-ReID, TransReID (trained on MSMT17)
- **Use case:** Testing generalization, domain adaptation research

**How it works:** Models use weights trained on the source domain and are evaluated on all target domains without retraining.

```bash
python run_benchmark.py --config benchmark/configs/cross_domain_test.yaml
```

---

### 5. `full_evaluation.yaml` ⭐ NEW
**Purpose:** Complete evaluation - Cross-domain + Zero-shot on all datasets

**What it runs:**
- **Datasets:** All 12 datasets (Market-1501, DukeMTMC, CUHK03, MSMT17, GRID, iLIDS, CelebReID, PKU, LASTID, iLIDS-VID, G2A, IUST-ReID)
- **Cross-domain models:** CLIP-ReID (2 variants), TransReID trained on MSMT17
- **Zero-shot models:** OSNet variants, CLIP variants, DINOv2, DINOv3, SigLIP2, PE-Core
- **Use case:** Comprehensive benchmark across all available models and datasets

**Estimated runtime:** 12-24 hours (192 model-dataset combinations)

```bash
python run_benchmark.py --config benchmark/configs/full_evaluation.yaml
```

**Features:**
- Tests generalization of MSMT17-trained models across all domains
- Evaluates zero-shot performance on diverse datasets
- Comprehensive comparison of all model families
- Includes both standard and challenging datasets

---

### 6. `minimal_template.yaml`
**Purpose:** Starting point for creating custom configurations.

**What it contains:**
- Basic structure with comments
- Simple model examples
- Commented advanced model examples

**Use case:** Copy and modify for your experiments

```bash
cp benchmark/configs/minimal_template.yaml my_experiment.yaml
# Edit my_experiment.yaml
python run_benchmark.py --config my_experiment.yaml
```

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
reid config show quick_validation
reid config show comprehensive_benchmark

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
# Enable cross-domain mode
cross_domain_mode: true

# Source domain (where model was trained)
source_domain: "market1501"

# Target domains (where to evaluate)
target_domains:
  - market1501    # In-domain baseline
  - dukemtmcreid  # Cross-domain
  - cuhk03        # Cross-domain
  - msmt17        # Cross-domain

# Models use source_domain weights
models:
  - type: "clipreid"
    name: "ViT-B-16"
    source_domain: "msmt17"  # Override to use MSMT17 weights
    pretrained_path: null
    # ... other params
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
- `dinov2` - Self-supervised ViT (v2)
- `dinov3` - Self-supervised ViT (v3)
- `siglip2` - Sigmoid loss CLIP variant

---

## Tips and Best Practices

### 1. Start Small
Begin with `quick_validation.yaml` to verify everything works before running large benchmarks.

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
