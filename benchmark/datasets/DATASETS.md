# Available Datasets

Complete reference for all datasets supported by the Object Re-Identification Benchmark Suite.

## Table of Contents
- [Standard Datasets](#standard-datasets)
- [Extended Datasets](#extended-datasets)
- [Dataset Configuration](#dataset-configuration)
- [Usage Examples](#usage-examples)

---

## Standard Datasets

These datasets use the standard torchreid data loader.

| Dataset Name | Identities | Images | Image Size (HxW) | Batch Size | Description |
|-------------|-----------|--------|------------------|------------|-------------|
| **market1501** | 1,501 | 32,668 | 256x128 | 100 | Market-1501 dataset, most commonly used benchmark |
| **dukemtmcreid** | 1,404 | 36,411 | 256x128 | 100 | DukeMTMC-reID dataset, second most popular |
| **cuhk03** | 1,467 | 28,192 | 256x128 | 100 | CUHK03 dataset with labeled/detected annotations |
| **grid** | 250 | 1,275 | 256x128 | 100 | GRID dataset, small for quick testing |
| **ilids** | 119 | 476 | 256x128 | 100 | iLIDS dataset, challenging conditions |

### Market-1501
- **Size**: 1,501 identities, 32,668 images
- **Cameras**: 6 cameras
- **Split**: 12,936 training + 19,732 testing images
- **Setting**: Outdoor market surveillance
- **Challenges**: Illumination, pose variation, occlusion
- **Use Case**: Primary benchmark for person re-identification

### DukeMTMC-reID
- **Size**: 1,404 identities, 36,411 images
- **Cameras**: 8 cameras
- **Split**: 16,522 training + 19,889 testing images
- **Setting**: Campus surveillance
- **Challenges**: Large viewpoint changes, complex backgrounds
- **Use Case**: Standard benchmark, often used with Market-1501

### CUHK03
- **Size**: 1,467 identities, 28,192 images
- **Cameras**: 10 camera pairs
- **Split**: Multiple splits available (split_id parameter)
- **Annotations**: Two types - labeled (manual) and detected (DPM detector)
- **Setting**: Campus surveillance
- **Special**: Uses CUHK03-specific evaluation metric
- **Configuration**:
  ```yaml
  cuhk03_labeled: true        # Use manual annotations
  cuhk03_classic_split: false # Use new protocol
  use_cuhk03_metric: true     # Use CUHK03 evaluation
  ```

### GRID
- **Size**: 250 identities, 1,275 images
- **Cameras**: 8 cameras
- **Split**: 125 training + 125 testing identities
- **Setting**: Underground station
- **Challenges**: Low resolution, severe occlusion
- **Use Case**: Quick testing, challenging benchmark

### iLIDS
- **Size**: 119 identities, 476 images
- **Cameras**: 2 cameras
- **Split**: Random splits (multiple runs recommended)
- **Setting**: Airport arrival hall
- **Challenges**: Occlusion, illumination changes
- **Use Case**: Small-scale benchmark

---

## Extended Datasets

These datasets use custom data loaders for specialized formats or requirements.

| Dataset Name | Type | Image Size (HxW) | Description | Custom Loader |
|-------------|------|------------------|-------------|---------------|
| **msmt17** | Large-scale | 256x128 | Multi-scene multi-time dataset | MSMT17 |
| **celebreid** | Celebrity | 256x128 | Celebrity re-identification | CelebReID |
| **pku** | Surveillance | 128x48 | Campus surveillance (unusual aspect ratio) | PKU |
| **lastid** | Airport | 256x128 | Large-scale airport surveillance | LASTID |
| **ilidsvid** | Video | 256x128 | Video-based re-identification | ILIDSVIDCustom |
| **g2a** | Gallery | 256x128 | Gallery-to-Archive dataset | G2AVReID |
| **iustreid** | Multi-camera | 256x128 | Multi-camera surveillance dataset | IUSTReID |

### MSMT17
- **Full Name**: Multi-Scene Multi-Time
- **Size**: 4,101 identities, 126,441 images
- **Cameras**: 15 cameras
- **Setting**: Multiple outdoor/indoor scenes, different times
- **Challenges**: Large scale, diverse scenarios, weather/time variations
- **Use Case**: Large-scale benchmark, cross-domain source dataset
- **Special**: Best for training models for generalization

### CelebrityReID
- **Size**: Varies (celebrity images)
- **Setting**: Celebrity photos from various sources
- **Challenges**: Clothing changes, pose variation, different contexts
- **Use Case**: Cross-domain to surveillance, celebrity tracking
- **Special**: Tests generalization to different image types

### PKU
- **Full Name**: PKU-Reid (Version 1a)
- **Size**: Variable
- **Image Size**: 128x48 (unusual aspect ratio)
- **Setting**: Campus surveillance
- **Challenges**: Low resolution, unusual aspect ratio
- **Use Case**: Low-resolution re-identification research
- **Special**: Requires models to handle different aspect ratios

### LASTID
- **Full Name**: Large-Scale Airport Surveillance Tracking
- **Setting**: Airport surveillance
- **Challenges**: Large scale, long-term tracking
- **Use Case**: Airport security, long-term re-identification
- **Special**: Real-world airport deployment scenario

### iLIDS-VID
- **Full Name**: iLIDS Video Dataset
- **Type**: Video sequences
- **Setting**: Airport arrival hall (video version of iLIDS)
- **Challenges**: Video-based matching, temporal information
- **Use Case**: Video re-identification, sequence matching
- **Special**: Tests temporal feature extraction

### G2A
- **Full Name**: Gallery-to-Archive
- **Setting**: Gallery matching to archive
- **Challenges**: Cross-setting matching
- **Use Case**: Gallery-based re-identification
- **Special**: Different query and gallery settings

### IUST-ReID
- **Setting**: Multi-camera surveillance system
- **Challenges**: Multi-camera consistency
- **Use Case**: Multi-camera tracking systems
- **Special**: Tests camera-invariant features

---

## Dataset Configuration

### Basic Configuration

Each dataset in the registry has these core parameters:

```python
@dataclass
class DatasetConfig:
    name: str                    # Dataset identifier
    source: str                  # Source dataset name for torchreid
    target: str                  # Target dataset name for torchreid
    height: int = 256           # Image height after resize
    width: int = 128            # Image width after resize
    batch_size: int = 100       # Test batch size
    split_id: int = 0           # Dataset split identifier
    use_cuhk03_metric: bool = False      # Use CUHK03 evaluation
    cuhk03_labeled: bool = True          # Use labeled annotations
    cuhk03_classic_split: bool = False   # Use classic split
    is_custom: bool = False              # Uses custom loader
    custom_class: Optional[type] = None  # Custom dataset class
```

### Configuration in YAML

```yaml
datasets:
  - market1501      # Simple name reference
  - dukemtmcreid
  - cuhk03
  - msmt17
```

### Advanced Configuration Options

For datasets requiring special handling:

```yaml
# CUHK03 with specific options
datasets:
  - name: cuhk03
    cuhk03_labeled: false      # Use detected annotations
    cuhk03_classic_split: true # Use classic split
    split_id: 0

# Custom batch size for GPU memory constraints
datasets:
  - name: msmt17
    batch_size: 64  # Reduce from default 100
```

---

## Dataset Statistics

### Query and Gallery Splits

| Dataset | Query Images | Gallery Images | Train Images | Distractors |
|---------|-------------|----------------|--------------|-------------|
| Market-1501 | 3,368 | 15,913 | 12,936 | Yes |
| DukeMTMC | 2,228 | 17,661 | 16,522 | Yes |
| CUHK03 | ~1,400 | ~5,328 | ~7,368 | No |
| GRID | 125 | 900 | 125 | Yes |
| iLIDS | ~119 | ~119 | - | No |
| MSMT17 | 11,659 | 82,161 | 32,621 | Yes |

### Camera Distribution

| Dataset | Number of Cameras | Camera Types |
|---------|------------------|--------------|
| Market-1501 | 6 | Fixed outdoor cameras |
| DukeMTMC | 8 | Fixed outdoor cameras |
| CUHK03 | 10 pairs | Indoor/outdoor paired cameras |
| GRID | 8 | Fixed underground cameras |
| iLIDS | 2 | Fixed indoor cameras |
| MSMT17 | 15 | Mixed indoor/outdoor cameras |

---

## Dataset Directory Structure

Expected directory structure in `reid-data/`:

```
reid-data/
├── market1501/
│   ├── bounding_box_train/
│   ├── bounding_box_test/
│   ├── query/
│   └── gt_bbox/
├── dukemtmcreid/
│   ├── bounding_box_train/
│   ├── bounding_box_test/
│   └── query/
├── cuhk03/
│   ├── cuhk03_release/
│   │   ├── cuhk-03.mat
│   │   └── ...
│   └── ...
├── msmt17/
│   ├── train/
│   ├── test/
│   └── list_*.txt
└── ...
```

---

## Usage Examples

### List Available Datasets

```bash
reid dataset list
```

### Check Dataset Download Status

```bash
reid dataset check
```

Output:
```
Dataset Status:
✓ market1501     - Available
✓ dukemtmcreid   - Available
✗ cuhk03         - Not found
✓ msmt17         - Available
```

### Get Dataset Information

```bash
reid dataset info market1501
```

### View Dataset Statistics

```bash
reid dataset stats market1501
```

### Configuration Example

```yaml
gpu_id: 0

data:
  root: "reid-data"

datasets:
  - market1501
  - dukemtmcreid
  - msmt17

models:
  - type: "osnet"
    name: "osnet_x1_0"

output:
  results_dir: "results"
  csv_filename: "results.csv"
```

### Cross-Domain Configuration

```yaml
cross_domain_mode: true
source_domain: "msmt17"  # Large diverse dataset for training

target_domains:
  - market1501  # Test generalization to other datasets
  - dukemtmcreid
  - cuhk03

data:
  root: "reid-data"

models:
  - type: "clipreid"
    name: "ViT-B-16"
    source_domain: "msmt17"
```

---

## Dataset Selection Guidelines

### For Quick Testing
- **GRID** - Small, fast
- **iLIDS** - Very small, quick results

### For Standard Benchmarking
- **Market-1501** - Most common, standard splits
- **DukeMTMC-reID** - Standard complement to Market-1501
- **CUHK03** - Standard third benchmark

### For Large-Scale Testing
- **MSMT17** - Large scale, diverse scenarios

### For Cross-Domain Training
- **MSMT17** as source - Most diverse, best generalization
- Test on Market-1501, DukeMTMC, CUHK03

### For Specialized Research
- **PKU** - Low resolution
- **iLIDS-VID** - Video sequences
- **CelebrityReID** - Celebrity images
- **LASTID** - Airport surveillance

---

## Common Issues and Solutions

### Dataset Not Found

```bash
# Check if dataset is in registry
reid dataset list

# Verify dataset directory exists
ls reid-data/market1501

# Check dataset configuration
reid dataset info market1501
```

### Empty Dataset Splits

**Problem**: "Empty dataset splits - Query: 0, Gallery: 0"

**Solutions**:
1. Verify dataset downloaded correctly
2. Check directory structure matches expected format
3. Ensure split_id is valid for the dataset

### Custom Dataset Loading Errors

**Problem**: "Failed to instantiate custom dataset class"

**Solutions**:
1. Check custom dataset class is properly imported
2. Verify dataset files exist in correct location
3. Check dataset format matches expected structure

### CUHK03 Metric Issues

**Problem**: Different results than reported in papers

**Solutions**:
1. Set `use_cuhk03_metric: true` in config
2. Check `cuhk03_labeled` vs `cuhk03_classic_split` settings
3. Use correct protocol (new protocol is default)

---

## Adding New Datasets

### Step 1: Create Custom Dataset Class

```python
# benchmark/datasets/custom/mydataset.py
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.train = self._load_train()
        self.query = self._load_query()
        self.gallery = self._load_gallery()

    def _load_train(self):
        # Return list of (img_path, pid, camid)
        pass

    def _load_query(self):
        pass

    def _load_gallery(self):
        pass
```

### Step 2: Register Dataset

```python
# benchmark/datasets/registry.py
from benchmark.datasets.custom.mydataset import MyDataset

DATASET_REGISTRY["mydataset"] = DatasetConfig(
    name="mydataset",
    source="mydataset",
    target="mydataset",
    height=256,
    width=128,
    batch_size=100,
    split_id=0,
    is_custom=True,
    custom_class=MyDataset,
)
```

### Step 3: Use in Configuration

```yaml
datasets:
  - mydataset
```

---

## References

### Papers

- **Market-1501**: Zheng et al., "Scalable Person Re-identification: A Benchmark", ICCV 2015
- **DukeMTMC-reID**: Ristani et al., "Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking", ECCVW 2016
- **CUHK03**: Li et al., "DeepReID: Deep Filter Pairing Neural Network for Person Re-identification", CVPR 2014
- **MSMT17**: Wei et al., "Person Transfer GAN to Bridge Domain Gap for Person Re-Identification", CVPR 2018

### Dataset Links

- Market-1501: http://www.liangzheng.org/Project/project_reid.html
- DukeMTMC-reID: https://github.com/layumi/DukeMTMC-reID_evaluation
- CUHK03: http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html
- MSMT17: https://www.pkuvmc.com/dataset.html

---

Last Updated: 2026-01-06
