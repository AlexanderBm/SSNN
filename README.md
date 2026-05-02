# Summary-Statistics Neural Network (SSNN)

A neural network training framework that operates entirely on GWAS summary
statistics, without individual-level genotype data.

## Setup

```bash
pip install -e ".[dev]"
```

## Running tests

```bash
pytest
```

## Project structure

- `src/ssnn/` -- core library (activations, Gaussian integrals, population risk, optimizer)
- `tests/` -- numerical validation of every mathematical claim
- `plan/` -- research plan (LaTeX)
