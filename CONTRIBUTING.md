# Contributing to XSearch

First off, thank you for considering contributing to XSearch! It's people like you that make open source such a great community. We welcome all contributions, from bug fixes to new features and documentation improvements.

## Getting Started

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/XSearch.git
   cd XSearch
   ```
3. **Set up your environment** (We recommend using Conda or a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```

## Development Workflow

1. **Create a branch:** Always create a new branch for your feature or bug fix from the `main` branch.
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-number
   ```
2. **Make your changes:**
   - Follow the PEP 8 style guide for Python code.
   - Add comments to complex logic.
3. **Test your changes:** Ensure the core pipelines work before submitting.
   - Run the training/evaluation pipeline (e.g., `python run.py ...`).
   - Run the retrieval evaluation (e.g., `python eval_retrieval.py ...`).
4. **Commit your changes:** Write clear and descriptive commit messages. For example: `feat: add cross-sample contrastive loss` or `fix: resolve dataloader index out of bounds`.
5. **Push and create a Pull Request (PR):** Push your branch to your fork and submit a PR to the original repository.

## Code Review Process

Once you open a PR, the maintainers will review your code. We might request some changes or optimizations. Once approved, your PR will be merged. Thank you for your contribution!