# Contributing to PyCausalImpact

Thanks for your interest in contributing!

## Development workflow
1. Fork and clone the repository.
2. Create a virtual environment and activate it.
3. Install dependencies with `pip install -e .[dev,test]`.
4. Install pre-commit hooks with `pre-commit install`.
5. Create a feature branch, make your changes, and run the checks below.
6. Open a pull request.

## Testing
Run the test suite to ensure everything works:

```bash
pytest
```

## Pre-commit
Format and lint your changes before committing:

```bash
pre-commit run --files <file1> [<file2> ...]
# or run on the entire repo
pre-commit run --all-files
```

## Coding style
- Code is formatted with **Black** (line length 88).
- Linting is enforced with **Flake8**.
- Keep functions and modules small and focused.

Following these steps helps keep PyCausalImpact reliable and easy to maintain. Happy hacking!
