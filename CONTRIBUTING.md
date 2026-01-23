# Contributing to Object Re-Identification Benchmark Suite

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/moiiai-tech/object-reid.git
cd object-reid

# Install dependencies including dev tools
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

## Code Style

- Follow PEP 8 conventions
- Use `ruff` for linting: `uv run ruff check .`
- Use `mypy` for type checking: `uv run mypy .`
- Keep comments minimal and only when necessary
- Follow DRY (Don't Repeat Yourself) and SOLID principles

## Running Tests

```bash
uv run pytest
uv run pytest --cov=benchmark --cov=reid
```

## Making Changes

1. Create a branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following the code style guidelines

3. Run linting and tests:
   ```bash
   uv run ruff check .
   uv run pytest
   ```

4. Commit with a descriptive message:
   ```bash
   git commit -m "feat: Add support for new model X"
   ```

## Commit Message Format

Use conventional commit prefixes:
- `feat:` New features
- `fix:` Bug fixes
- `refactor:` Code refactoring
- `docs:` Documentation changes
- `test:` Adding or updating tests

## Adding New Models

1. Create a wrapper in `benchmark/models/` following the base class pattern
2. Register the model in `benchmark/models/factory.py`
3. Add documentation to `benchmark/models/MODELS.md`
4. Add tests for the new model

## Adding New Datasets

1. Create a loader in `benchmark/datasets/custom/` if needed
2. Register the dataset in `benchmark/datasets/registry.py`
3. Add documentation to `benchmark/datasets/DATASETS.md`
4. Update DVC configuration if applicable

## Pull Request Process

1. Ensure all tests pass
2. Update documentation if needed
3. Describe your changes in the PR description
4. Request review from maintainers

## Questions

Open an issue on GitHub for questions or discussions.
