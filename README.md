# Object Re-Identification Benchmark Suite

A config-driven framework for benchmarking Person Re-Identification models across multiple datasets.

## What is this?

This framework allows you to evaluate and compare different re-identification models (OSNet, CLIP, CLIP-ReID, TransReID, DINOv2, etc.) across standard datasets (Market-1501, DukeMTMC, MSMT17, etc.) using simple YAML configurations.

**Key Features:**
- 8 model families with 25+ variants
- 12+ standard ReID datasets
- YAML-based configuration
- Automatic weight download
- Cross-domain evaluation
- Rich CLI interface

## Installation

### Using uv (recommended)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/moiiai-tech/object-reid.git
cd object-reid
uv sync
```

### Using pip

```bash
git clone https://github.com/moiiai-tech/object-reid.git
cd object-reid
pip install -e .
```

### External Dependencies

Some models require external repositories. Create the `external/` folder and clone the required dependencies:

```bash
mkdir -p external
cd external

# CLIP-ReID (required for clipreid model)
git clone https://github.com/Syliz517/CLIP-ReID.git

# TransReID (required for transreid model)
git clone https://github.com/damo-cv/TransReID.git

cd ..
```

See [external/README.md](external/README.md) for more details.

### Verify Installation

```bash
reid info
```

## Quick Start

### 1. Run a Quick Test

```bash
reid benchmark quick
```

This runs OSNet on Market-1501 in ~5-10 minutes to verify your setup.

### 2. Create a Configuration

**Interactive mode:**
```bash
reid config create
```

**Or manually create a YAML file:**
```yaml
gpu_id: 0

data:
  root: "reid-data"

datasets:
  - market1501
  - dukemtmcreid

models:
  - type: "osnet"
    name: "osnet_x1_0"

  - type: "clipreid"
    name: "ViT-B-16"
    stride_size: [16, 16]
    sie_camera: true

output:
  results_dir: "results"
  csv_filename: "benchmark_results.csv"
```

### 3. Run Your Benchmark

```bash
reid benchmark run --config my_config.yaml
```

Or:
```bash
python run_benchmark.py --config my_config.yaml
```

### 4. View Results

```bash
reid results show benchmark_results.csv
```

## CLI Commands

### Dataset Commands
```bash
reid dataset list          # List all datasets
reid dataset check         # Check download status
reid dataset info market1501  # Dataset details
```

### Model Commands
```bash
reid model list            # List all models
reid model info clipreid   # Model details
```

### Config Commands
```bash
reid config list           # List available configs
reid config validate my_config.yaml  # Validate config
reid config preview my_config.yaml   # Preview execution
reid config templates      # List config templates
```

### Benchmark Commands
```bash
reid benchmark quick       # Quick validation test
reid benchmark run --config my_config.yaml  # Run benchmark
```

### Results Commands
```bash
reid results list          # List result files
reid results show results.csv  # Display results
reid results compare run1.csv run2.csv  # Compare runs
```

## Available Configurations

Pre-built configuration templates in `benchmark/configs/`:

- `quick_validation.yaml` - Fast smoke test
- `comprehensive_benchmark.yaml` - Full evaluation (4 datasets, 5 models)
- `advanced_models.yaml` - Compare modern architectures
- `cross_domain_test.yaml` - Cross-domain evaluation

View all templates:
```bash
reid config templates
```

## Documentation

**Detailed References:**
- [Models Reference](benchmark/models/MODELS.md) - All 25+ model variants, parameters, and configurations
- [Datasets Reference](benchmark/datasets/DATASETS.md) - All 12 datasets with statistics and usage guidelines

**Quick Info:**
```bash
reid model info <model_type>    # Get model details
reid dataset info <dataset_name>  # Get dataset details
```

## Example Workflows

### Quick Test
```bash
reid benchmark quick
```

### Comprehensive Benchmark
```bash
reid benchmark run --config benchmark/configs/comprehensive_benchmark.yaml
```

### Custom Benchmark
```bash
# Create config
reid config create

# Validate
reid config validate my_config.yaml --strict

# Run
reid benchmark run --config my_config.yaml

# View results
reid results show results.csv
```

### Cross-Domain Evaluation
```yaml
cross_domain_mode: true
source_domain: "msmt17"

target_domains:
  - market1501
  - dukemtmcreid

models:
  - type: "clipreid"
    name: "ViT-B-16"
    source_domain: "msmt17"
```

## Supported Models

| Family | Variants | Type |
|--------|----------|------|
| OSNet | x1_0, x0_75, x0_5, x0_25 | CNN |
| CLIP | ViT-B/32, ViT-B/16, ViT-L/14 | Vision-Language |
| CLIP-ReID | ViT-B-16 | Fine-tuned |
| TransReID | vit_base_patch16_224 | Transformer |
| PE-Core | PE-Core-L14-336 | Large Vision |
| DINOv2 | vitb14, vitl14, vitg14 | Self-supervised |
| DINOv3 | vitb14, vitl14 | Self-supervised |
| SigLIP2 | base-patch16-256/384 | Vision-Language |

See [Models Reference](benchmark/models/MODELS.md) for detailed specifications.

## Supported Datasets

**Standard:** Market-1501, DukeMTMC-reID, CUHK03, GRID, iLIDS
**Extended:** MSMT17, CelebrityReID, PKU, LASTID, iLIDS-VID, G2A, IUST-ReID

See [Datasets Reference](benchmark/datasets/DATASETS.md) for detailed information.

## Project Structure

```
object-reid/
├── reid/                   # CLI interface
├── benchmark/              # Core benchmark logic
│   ├── datasets/          # Dataset loaders and registry
│   ├── models/            # Model wrappers and factory
│   ├── utils/             # Utilities and helpers
│   └── configs/           # Configuration templates
├── external/               # External model implementations
├── run_benchmark.py        # Main entry point
└── pyproject.toml         # Project configuration
```

## Troubleshooting

### CUDA Out of Memory
Reduce batch size in your config:
```yaml
datasets:
  - name: market1501
    batch_size: 64  # Reduce from default 100
```

### Dataset Not Found
Check dataset availability:
```bash
reid dataset check
```

### Model Weights Not Loading
Check weight status:
```bash
reid model weights
```

Or specify path explicitly:
```yaml
models:
  - type: "clipreid"
    pretrained_path: "/path/to/weights.pth"
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute.
