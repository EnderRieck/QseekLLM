# Contributing to LLMTrain

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/llmTrain.git
cd llmTrain

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in editable mode with dev dependencies
pip install -e '.[dev,eval]'

# Run tests
pytest
```

## Code Style

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Add docstrings for public functions and classes
- Keep lines under 120 characters
- Use `black` for code formatting (if available)

## Project Structure

- `src/llmtrain/`: Core library code
  - `models/`: Model architectures
  - `data/`: Data loading and preprocessing
  - `training/`: Training loop and optimization
  - `evaluation/`: Validation and evaluation
  - `checkpointing/`: Checkpoint management
  - `distributed/`: FSDP/DDP utilities
- `configs/`: YAML configuration files
- `tools/`: Standalone scripts
- `scripts/`: Utility bash scripts
- `tests/`: Unit and integration tests

## Adding New Features

### Adding a New Model Architecture

1. Create `src/llmtrain/models/your_model.py`
2. Implement the model class inheriting from `nn.Module`
3. Register in `src/llmtrain/models/__init__.py`
4. Add config in `configs/model/your_model.yaml`
5. Add tests in `tests/models/test_your_model.py`

### Adding a New Data Source

1. Implement parser in `src/llmtrain/preprocessing/parsers/`
2. Register in `src/llmtrain/preprocessing/parsers/__init__.py`
3. Add config schema in `src/llmtrain/preprocessing/config.py`
4. Add example config in `configs/preprocess/`
5. Add tests

### Adding a New Scheduler

1. Implement in `src/llmtrain/training/scheduler.py`
2. Register in `get_scheduler()` factory function
3. Add config schema in `src/llmtrain/training/config.py`
4. Add tests

## Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/data/test_packer.py

# Run with coverage
pytest --cov=llmtrain --cov-report=html
```

## Pull Request Process

1. **Fork** the repository
2. **Create a branch** for your feature: `git checkout -b feature/your-feature-name`
3. **Make your changes** with clear, atomic commits
4. **Add tests** for new functionality
5. **Run tests** to ensure nothing breaks
6. **Update documentation** if needed
7. **Push** to your fork: `git push origin feature/your-feature-name`
8. **Open a Pull Request** with a clear description

### PR Guidelines

- Keep PRs focused on a single feature or fix
- Write clear commit messages
- Reference related issues in the PR description
- Ensure all tests pass
- Update README/docs if adding user-facing features

## Commit Message Format

```
<type>: <short summary>

<optional detailed description>

<optional footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `perf`: Performance improvements
- `chore`: Maintenance tasks

Example:
```
feat: add cosine annealing scheduler

Implements cosine annealing with warmup for learning rate scheduling.
Includes config schema and tests.

Closes #42
```

## Reporting Issues

When reporting bugs, please include:
- Python version and OS
- PyTorch version
- Full error traceback
- Minimal reproducible example
- Config file (if applicable)

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas
- Check existing issues before creating new ones

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
