# External Dependencies

This directory contains external model repositories and dependencies used in the benchmark framework.

## Structure

```
external/
├── CLIP-ReID/           # CLIP-ReID model repository
│   └── ...              # Vision-language model for person re-identification
│
├── TransReID/           # TransReID model repository
│   └── ...              # Transformer-based object re-identification
│
└── perception_models/   # PE-Core perception models
    └── core/            # Core vision encoder components
        └── vision_encoder/
            ├── pe.py              # PE-Core CLIP implementation
            ├── transforms.py      # Image transformations
            ├── rope.py            # Rotary Position Embeddings
            └── ...
```

## Models

### CLIP-ReID
- **Repository**: https://github.com/Syliz517/CLIP-ReID
- **Paper**: "CLIP-ReID: Exploiting Vision-Language Model for Image Re-Identification without Concrete Text Labels"
- **Variants**: ViT-B-16, RN50
- **Integration**: See [benchmark/models/README_CLIPREID.md](../benchmark/models/README_CLIPREID.md)

### TransReID
- **Repository**: https://github.com/damo-cv/TransReID
- **Paper**: "TransReID: Transformer-based Object Re-Identification" (ICCV 2021)
- **Variants**: ViT-Base, ViT-Small, DeiT-Small (with SIE and JPM modules)
- **Integration**: See [benchmark/models/README_TRANSREID.md](../benchmark/models/README_TRANSREID.md)

### PE-Core (Perception Models)
- **Source**: Custom perception models library
- **Components**: Vision encoders with CLIP-based architecture
- **Models**: PE-Core-L14-336 and variants
- **Integration**: See [benchmark/models/pecore.py](../benchmark/models/pecore.py)

## Usage

These external dependencies are automatically added to the Python path by the model wrappers:

- **CLIP-ReID**: Added in [clipreid_wrapper.py](../benchmark/models/clipreid_wrapper.py)
- **TransReID**: Added in [transreid_wrapper.py](../benchmark/models/transreid_wrapper.py)
- **PE-Core**: Added in [pecore.py](../benchmark/models/pecore.py) and [run_benchmark.py](../run_benchmark.py)

No manual path configuration is required.

## Gitignore

All external model repositories are included in `.gitignore` to avoid committing large codebases:
- `external/CLIP-ReID/`
- `external/TransReID/`
- `external/perception_models/`

Make sure to clone/download these dependencies before running benchmarks.
