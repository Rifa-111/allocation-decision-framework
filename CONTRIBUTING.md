# Contributing to Allocation Decision Framework

Thank you for your interest in contributing! This document outlines how to contribute effectively.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/allocation-decision-framework.git
   cd allocation-decision-framework
   ```
3. **Create a development branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Workflow

### Code Style

- **Format code** with Black:
  ```bash
  black allocation_framework/
  ```
- **Lint** with Flake8:
  ```bash
  flake8 allocation_framework/
  ```
- **Type hints** are required. Check with mypy:
  ```bash
  mypy allocation_framework/
  ```

### Testing

- Write tests for all new features in `tests/`
- Run tests locally:
  ```bash
  pytest tests/ -v --cov=allocation_framework
  ```
- Aim for >80% code coverage

### Documentation

- Document public functions and classes with docstrings
- Update README.md and relevant docs/ files
- Add examples or notebooks demonstrating new features

## Contribution Types

### Bug Reports

File an issue with:
- Clear description of the bug
- Steps to reproduce
- Expected vs. actual behavior
- Python version & environment info

### Feature Requests

Propose new features with:
- Clear use case and motivation
- How it aligns with the framework's goals
- Potential implementation approach

### Code Contributions

1. **Ensure tests pass**:
   ```bash
   pytest tests/ --cov=allocation_framework
   ```
2. **Update docstrings** and type hints
3. **Add examples** if applicable
4. **Reference the paper** if feature builds on it
5. **Submit a pull request** with a clear description

### Documentation

- Fix typos and improve clarity
- Add tutorials or examples
- Improve API documentation
- Update ARCHITECTURE.md if needed

## Pull Request Process

1. **Update your branch** with latest main:
   ```bash
   git pull origin main
   ```
2. **Write descriptive commit messages**:
   ```
   [feature] Add allocation-aware loss function
   
   Implements the fairness-weighted loss from the paper's
   Section 3.2. Includes unit tests and example notebook.
   ```
3. **Push and create a PR** on GitHub
4. **Link related issues** (e.g., "Closes #42")
5. **Address review feedback** promptly

## Code of Conduct

- Be respectful and inclusive
- Assume good intent
- Focus on ideas, not people
- Report conduct issues to the maintainer

## Questions?

- Open a GitHub issue with the `question` label
- Check existing issues first
- Be specific about what you're trying to achieve

## Recognition

Contributors will be acknowledged in:
- CONTRIBUTORS.md file
- Release notes for their contributions

Thank you for helping make allocation-aware healthcare AI more accessible! 🙏
